use gam::families::survival::location_scale::ResidualDistribution;
use gam::families::survival::lognormal_kernel::FrailtySpec;
// `predictor()` is provided by the predict extension trait,
// which moved into `gam-predict` when the prediction engine was peeled out.
use gam::inference::model::{
    FittedEstimator, FittedFamily, FittedModel, FittedModelPayload, MODEL_PAYLOAD_VERSION,
    ModelKind, PredictModelClass, SAVED_MODEL_KIND,
};
use gam::solver::estimate::{
    BlockRole, FitArtifacts, FittedBlock, FittedLinkState, UnifiedFitResult, UnifiedFitResultParts,
};
use gam::solver::pirls::PirlsStatus;
use gam::types::{
    InverseLink, LatentCLogLogState, LikelihoodScaleMetadata, LikelihoodSpec,
    LogLikelihoodNormalization, ResponseFamily, StandardLink,
};
use gam_predict::FittedModelPredictExt;
use ndarray::{Array1, Array2};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use tempfile::tempdir;

const EXPECTED_SAVED_MODEL_ROOT_FIELD_COUNT: usize = 3;
// Any payload-field or skip-rule change requires a fresh enumeration and a
// stateful-sync audit before the key-count pin below changes: the count is
// what detects a payload field the stateful sync has not been audited for.
// Keep it a literal. A count derived from the struct would pass whenever a field
// is added, and this failure is what forces the audit, so the literal is the
// control.
//
// The schema version is not pinned. The assert reads `MODEL_PAYLOAD_VERSION`,
// so a saved model must record the version this binary writes. A literal copy
// of the constant stayed at 16 through the bumps to 17, 18 and 19 while no
// gate ran this binary. A bump that adds no payload key changes what a field
// means, and `inference/model.rs` documents that beside the constant.
//
// Enumerated at schema version 22 (i2939's enumeration at 19, re-counted by
// i2955 at 20 and by gam#2926 at 22): `FittedModelPayload` declares 104 `pub`
// fields and has no `#[serde(skip)]` and no `#[serde(flatten)]`. Four carry
// `skip_serializing_if`, and these fixtures leave all four at their skipped value:
// `declared_latent_law=None`, `declared_latent_law_compression=None`,
// `group_metadata=None` and `deployment_extensions=[]`. So the JSON payload
// carries 104 - 4 = 100 keys. The version 12 fitted-estimator tag is one of them:
// it keeps an expectile target from decoding as a Gaussian observation law.
//
// Stateful-sync audit for the fields added since the last correct pin (96 keys,
// schema 15): `basis_adequacy`, `declared_latent_law`,
// `declared_latent_law_compression`, `residual_repair`, `score_crossfit_folds`,
// `score_transform`, `slope_time_basis`,
// `survival_marginal_slope_joint_latent_law` and `unidentified_scalar_terms`,
// plus the `logslope` -> `slope` renames (`baseline_slope(s)`,
// `slope_formula(s)`, `resolved_slopespec(s)`). Four fields were removed:
// `adaptive_regularization_diagnostics`, `gaussian_jackknife_plus`,
// `survival_time_smooth_lambda` and `survivalridge_lambda`. None is a fitted
// link state. `score_transform` nests a CTN stage's transformation-normal
// payload (`inference/ctn.rs` `fit_chain`), a family shape with no
// stateful-link slot. `residual_repair` carries the Bernoulli marginal-slope
// residual block's geometry (gam#2924, `#[serde(default)]` only, so it always
// serializes). `FittedModel::synchronize_stateful_link_metadata` mirrors every
// `FittedLinkState` variant through an exhaustive match, so a new stateful link
// cannot be persisted without a sync arm. Schema 17 re-reads a beta-logistic
// `sas_state` as the standardized link, and it syncs through the `BetaLogistic`
// arm into the same `sas_state` slot this file pins for SAS. Schema 18 adds
// `rho_posterior` inside the fit artifacts, not a payload key. Schema 19 deletes
// `FitInference`'s four covariance/SE copies inside the fit result (#2955).
// Schema 20 adds `FitInference::edf_rank_bound` there (#2901). Neither is a
// payload key or a stateful-link slot.
// Schema 22 records the #2954 certificate's Newton polish and each railed coordinate's
// face kind inside the fit artifacts: neither is a payload key or a stateful-link slot.
// Schema 23 adds `latent_law_consumed`, which records the latent law a marginal-slope
// fit consumed and its certificate (gam#2926, `#[serde(default)]` only, so it always
// serializes); it is a fit record, not a fitted link state, so no stateful-link slot
// changes.
// Two keys landed after that count without re-counting it:
// `gaussian_sigma_floor` (the Gaussian location-scale sigma floor, a scalar) and
// `informational_notes` (the fit's record of engine-chosen defaults). Neither is
// a fitted link state. gam#4466 then removes six write-only keys:
// `sas_param_covariance` and `mixture_link_param_covariance` (second copies of
// the covariance that `fit_result.fitted_link` already serializes, which
// prediction reads), `beta_noise` (a copy of the fit's `Scale` block, which
// prediction reads), `slope_formulas` and `baseline_slopes` (singleton mirrors
// of `slope_formula` and `baseline_slope` that nothing read) and the
// never-written `latent_score_contract`, leaving 96 payload keys. Schema 37
// (gam#3350) then stores the fit once and moves the version out of the payload:
// the `unified` copy of `fit_result` is gone and the version lives in the
// saved-model envelope `{kind, version, model}`, so the payload carries
// 96 - 2 = 94 keys and the root 3. The stateful sync keeps mirroring each
// link's point state; the link covariance stays on the fit.
const EXPECTED_MODEL_PAYLOAD_FIELD_COUNT: usize = 94;
const EXPECTED_STANDARD_FAMILY_FIELD_COUNT: usize = 6;

fn read_saved_model_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read model"))
        .expect("saved model json")
}

fn assert_saved_model_schema_is_pinned(saved: &Value) {
    let root = saved.as_object().expect("saved model root object");
    assert_eq!(
        root.len(),
        EXPECTED_SAVED_MODEL_ROOT_FIELD_COUNT,
        "saved model root fields changed; audit enum envelope coverage before updating this test"
    );
    assert_eq!(
        saved.get("kind").and_then(Value::as_str),
        Some(SAVED_MODEL_KIND),
        "saved model envelope changed; audit stateful sync coverage before updating this test"
    );
    assert_eq!(
        saved.get("version").and_then(Value::as_u64),
        Some(u64::from(MODEL_PAYLOAD_VERSION)),
        "a saved model must record, once, the version this binary writes"
    );
    let payload = saved_model_payload(saved);
    assert_eq!(
        payload.len(),
        EXPECTED_MODEL_PAYLOAD_FIELD_COUNT,
        "saved model payload fields changed; audit stateful sync coverage before updating this test"
    );
    for duplicate in ["unified", "version", "model_type"] {
        assert!(
            payload.get(duplicate).is_none(),
            "a saved model records `{duplicate}` at most once, outside the model"
        );
    }
    if let Some(fit) = payload.get("fit_result").and_then(Value::as_object) {
        assert_eq!(
            fit.get("training_sample_size").and_then(Value::as_u64),
            Some(8),
            "fit_result must persist the authoritative training row count"
        );
    }
    assert_eq!(
        serde_json::from_value::<FittedEstimator>(
            payload
                .get("estimator")
                .expect("required estimator metadata")
                .clone()
        )
        .expect("decode estimator metadata"),
        FittedEstimator::Likelihood,
        "ordinary likelihood fits must persist their estimator identity"
    );
}

fn saved_model_payload(saved: &Value) -> &serde_json::Map<String, Value> {
    saved
        .get("model")
        .and_then(Value::as_object)
        .expect("saved model payload object")
}

fn standard_family_state(saved: &Value) -> &serde_json::Map<String, Value> {
    let payload = saved_model_payload(saved);
    let family_state = payload
        .get("family_state")
        .and_then(Value::as_object)
        .expect("family_state object");
    assert_eq!(
        family_state.len(),
        EXPECTED_STANDARD_FAMILY_FIELD_COUNT,
        "standard family_state fields changed; audit stateful sync coverage before updating this test"
    );
    assert_eq!(
        family_state.get("family_kind").and_then(Value::as_str),
        Some("standard")
    );
    family_state
}

/// The SAS link-parameter covariance round-trips through its canonical home,
/// `fit_result.fitted_link`, which is what link-uncertainty prediction reads.
fn assert_loaded_sas_covariance(payload: &FittedModelPayload, expected: &Array2<f64>) {
    let fit = payload
        .fit_result
        .as_ref()
        .expect("loaded standard model carries its canonical fit_result");
    let FittedLinkState::Sas { covariance, .. } = &fit.fitted_link else {
        panic!("expected a SAS fitted link, got {:?}", fit.fitted_link);
    };
    assert_eq!(covariance.as_ref(), Some(expected));
}

fn minimal_fit_result(fitted_link: FittedLinkState) -> UnifiedFitResult {
    UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks: vec![FittedBlock {
            beta: Array1::from_vec(vec![0.0]),
            role: BlockRole::Mean,
            edf: 0.0,
            lambdas: Array1::zeros(0),
        }],
        training_sample_size: 8,
        log_lambdas: Array1::zeros(0),
        lambdas: Array1::zeros(0),
        likelihood_family: Some(LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        )),
        likelihood_scale: LikelihoodScaleMetadata::ProfiledGaussian,
        log_likelihood_normalization: LogLikelihoodNormalization::Full,
        log_likelihood: 0.0,
        deviance: 0.0,
        reml_score: Some(0.0),
        stable_penalty_term: 0.0,
        penalized_objective: Some(0.0),
        used_device: false,
        outer_iterations: 1,
        outer_converged: true,
        outer_gradient_norm: None,
        standard_deviation: 1.0,
        covariance_conditional: Some(
            Array2::from_shape_vec((1, 1), vec![1.0e-3]).expect("1x1 covariance"),
        ),
        covariance_corrected: None,
        inference: None,
        fitted_link,
        geometry: None,
        block_states: Vec::new(),
        pirls_status: PirlsStatus::Converged,
        max_abs_eta: 0.0,
        constraint_kkt: None,
        artifacts: FitArtifacts {
            pirls: None,
            ..Default::default()
        },
        inner_cycles: 0,
    })
    .expect("minimal fit result must be valid")
}

fn minimal_survival_fit_result() -> UnifiedFitResult {
    UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks: vec![
            FittedBlock {
                beta: Array1::from_vec(vec![0.0]),
                role: BlockRole::Threshold,
                edf: 0.0,
                lambdas: Array1::zeros(0),
            },
            FittedBlock {
                beta: Array1::from_vec(vec![0.0]),
                role: BlockRole::Scale,
                edf: 0.0,
                lambdas: Array1::zeros(0),
            },
        ],
        training_sample_size: 8,
        log_lambdas: Array1::zeros(0),
        lambdas: Array1::zeros(0),
        likelihood_family: Some(LikelihoodSpec::new(
            ResponseFamily::RoystonParmar,
            InverseLink::Standard(StandardLink::Identity),
        )),
        likelihood_scale: LikelihoodScaleMetadata::Unspecified,
        log_likelihood_normalization: LogLikelihoodNormalization::Full,
        log_likelihood: 0.0,
        deviance: 0.0,
        reml_score: Some(0.0),
        stable_penalty_term: 0.0,
        penalized_objective: Some(0.0),
        used_device: false,
        outer_iterations: 1,
        outer_converged: true,
        outer_gradient_norm: None,
        standard_deviation: 1.0,
        covariance_conditional: Some(
            Array2::from_shape_vec((2, 2), vec![1.0e-3, 0.0, 0.0, 1.0e-3]).expect("2x2 covariance"),
        ),
        covariance_corrected: None,
        inference: None,
        fitted_link: FittedLinkState::Standard(None),
        geometry: None,
        block_states: Vec::new(),
        pirls_status: PirlsStatus::Converged,
        max_abs_eta: 0.0,
        constraint_kkt: None,
        artifacts: FitArtifacts {
            pirls: None,
            ..Default::default()
        },
        inner_cycles: 0,
    })
    .expect("minimal survival fit result must be valid")
}

fn minimal_standard_model_with_group_metadata(
    group_metadata: Option<BTreeMap<String, Value>>,
) -> FittedModel {
    let mut payload = FittedModelPayload::new(
        "y ~ group(g)".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Gaussian,
                InverseLink::Standard(StandardLink::Identity),
            ),
            link: Some(StandardLink::Identity),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "gaussian".to_string(),
    );
    payload.fit_result = Some(minimal_fit_result(FittedLinkState::Standard(None)));
    payload.data_schema = Some(gam::inference::model::DataSchema { columns: vec![] });
    payload.group_metadata = group_metadata;
    FittedModel::from_payload(payload)
}

#[test]
fn saved_model_group_metadata_is_optional_and_roundtrips() {
    let mut group_metadata = BTreeMap::new();
    group_metadata.insert(
        "alpha".to_string(),
        serde_json::json!({
            "source": "registry-a",
            "batch": 7,
            "scores": [0.25, 0.75],
            "audited": true
        }),
    );
    group_metadata.insert(
        "beta".to_string(),
        serde_json::json!({
            "source": "registry-b",
            "batch": 8,
            "tags": ["heldout", "priority"],
            "audited": false
        }),
    );

    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("group-metadata-model.json");
    let model = minimal_standard_model_with_group_metadata(Some(group_metadata.clone()));
    model.save_to_path(&path).expect("save model");

    let saved = read_saved_model_json(&path);
    assert_eq!(
        saved_model_payload(&saved).get("group_metadata"),
        Some(&serde_json::to_value(&group_metadata).expect("group metadata json"))
    );

    let loaded = FittedModel::load_from_path(&path).expect("load model");
    let FittedModel::Standard { payload } = loaded else {
        panic!("expected standard model");
    };
    assert_eq!(payload.group_metadata, Some(group_metadata));
}

#[test]
fn save_and_load_syncs_standard_sas_state_from_fit_result() {
    let log_delta = -0.4;
    let sas_state =
        gam::mixture_link::sas_link_state_from_raw(0.25, log_delta).expect("valid sas state");
    let covariance =
        Array2::from_shape_vec((2, 2), vec![0.1, 0.02, 0.02, 0.2]).expect("2x2 covariance");
    let mut payload = FittedModelPayload::new(
        "y ~ x".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Sas(sas_state)),
            link: None,
            latent_cloglog_state: None,
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
    payload.set_training_feature_metadata(vec!["x".to_string()], vec![(-1.5f64, 2.25f64)]);
    payload.resolved_termspec = Some(gam::terms::smooth::TermCollectionSpec {
        linear_terms: vec![],
        smooth_terms: vec![],
        random_effect_terms: vec![],
        level: Default::default(),
    });

    let model = FittedModel::from_payload(payload);
    let saved_state = model
        .saved_sas_state()
        .expect("saved sas state")
        .expect("expected synchronized sas state");
    assert_eq!(saved_state.epsilon, sas_state.epsilon);
    assert_eq!(saved_state.log_delta, sas_state.log_delta);
    assert_eq!(saved_state.delta, sas_state.delta);

    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("model.json");
    model.save_to_path(&path).expect("save model");

    let saved = read_saved_model_json(&path);
    assert_saved_model_schema_is_pinned(&saved);
    let family_state = standard_family_state(&saved);
    assert_eq!(
        family_state.get("likelihood"),
        Some(&serde_json::json!({
            "response": "Binomial",
            "link": {
                "Sas": sas_state
            }
        }))
    );
    // The coarse `link` tag is `Option<StandardLink>`, and `StandardLink` only
    // carries the stateless links (Logit/Probit/CLogLog/Identity/Log). SAS is a
    // parametric link whose full state lives in `likelihood.link = {"Sas": ..}`
    // and in the synchronized `sas_state` below, so it has no `StandardLink`
    // representation and the coarse tag stays null.
    assert_eq!(family_state.get("link"), Some(&Value::Null));
    assert_eq!(
        family_state.get("sas_state"),
        Some(&serde_json::to_value(sas_state).expect("sas state json")),
        "serialized model should include synchronized family_state.sas_state"
    );
    assert_eq!(family_state.get("latent_cloglog_state"), Some(&Value::Null));
    assert_eq!(family_state.get("mixture_state"), Some(&Value::Null));
    assert_eq!(
        saved_model_payload(&saved).get("training_headers"),
        Some(&serde_json::json!(["x"]))
    );
    assert_eq!(
        saved_model_payload(&saved).get("training_feature_ranges"),
        Some(&serde_json::json!([[-1.5, 2.25]]))
    );

    let loaded = FittedModel::load_from_path(&path).expect("load model");
    let loaded_state = loaded
        .saved_sas_state()
        .expect("loaded sas state")
        .expect("expected loaded sas state");
    assert_eq!(loaded_state.epsilon, sas_state.epsilon);
    assert_eq!(loaded_state.log_delta, sas_state.log_delta);
    assert_eq!(loaded_state.delta, sas_state.delta);
    let FittedModel::Standard { payload } = loaded else {
        panic!("expected standard model");
    };
    assert_loaded_sas_covariance(&payload, &covariance);
    assert_eq!(payload.training_headers, Some(vec!["x".to_string()]));
    assert_eq!(payload.training_feature_ranges, Some(vec![(-1.5, 2.25)]));
}

#[test]
fn save_and_load_syncs_standard_latent_cloglog_state_from_fit_result() {
    let latent_state = LatentCLogLogState::new(0.65).expect("valid latent state");
    let mut payload = FittedModelPayload::new(
        "y ~ x".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::LatentCLogLog(latent_state),
            ),
            link: Some(StandardLink::CLogLog),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "latent-cloglog-binomial".to_string(),
    );
    payload.fit_result = Some(minimal_fit_result(FittedLinkState::LatentCLogLog {
        state: latent_state,
    }));
    payload.data_schema = Some(gam::inference::model::DataSchema { columns: vec![] });
    payload.set_training_feature_metadata(
        vec!["x".to_string(), "z".to_string()],
        vec![(0.0f64, 1.0f64), (-3.5f64, 4.0f64)],
    );
    payload.resolved_termspec = Some(gam::terms::smooth::TermCollectionSpec {
        linear_terms: vec![],
        smooth_terms: vec![],
        random_effect_terms: vec![],
        level: Default::default(),
    });

    let model = FittedModel::from_payload(payload);
    let saved_state = model
        .saved_latent_cloglog_state()
        .expect("saved latent cloglog state")
        .expect("expected synchronized latent cloglog state");
    assert_eq!(saved_state, latent_state);
    assert_eq!(
        model
            .resolved_inverse_link()
            .expect("resolved inverse link"),
        Some(InverseLink::LatentCLogLog(latent_state))
    );

    let dir = tempdir().expect("temp dir");
    let path = dir.path().join("latent-cloglog-model.json");
    model.save_to_path(&path).expect("save model");

    let saved = read_saved_model_json(&path);
    assert_saved_model_schema_is_pinned(&saved);
    let family_state = standard_family_state(&saved);
    assert_eq!(
        family_state.get("likelihood"),
        Some(&serde_json::json!({
            "response": "Binomial",
            "link": {
                "LatentCLogLog": latent_state
            }
        }))
    );
    assert_eq!(
        family_state.get("link").and_then(Value::as_str),
        Some("CLogLog")
    );
    assert_eq!(
        family_state.get("latent_cloglog_state"),
        Some(&serde_json::to_value(latent_state).expect("latent state json")),
        "serialized model should include synchronized family_state.latent_cloglog_state"
    );
    assert_eq!(family_state.get("mixture_state"), Some(&Value::Null));
    assert_eq!(family_state.get("sas_state"), Some(&Value::Null));
    assert_eq!(
        saved_model_payload(&saved).get("training_headers"),
        Some(&serde_json::json!(["x", "z"]))
    );
    assert_eq!(
        saved_model_payload(&saved).get("training_feature_ranges"),
        Some(&serde_json::json!([[0.0, 1.0], [-3.5, 4.0]]))
    );

    let loaded = FittedModel::load_from_path(&path).expect("load model");
    let loaded_state = loaded
        .saved_latent_cloglog_state()
        .expect("loaded latent cloglog state")
        .expect("expected loaded latent cloglog state");
    assert_eq!(loaded_state, latent_state);
    assert_eq!(
        loaded
            .resolved_inverse_link()
            .expect("loaded resolved inverse link"),
        Some(InverseLink::LatentCLogLog(latent_state))
    );
    let FittedModel::Standard { payload } = loaded else {
        panic!("expected standard model");
    };
    assert_eq!(
        payload.training_headers,
        Some(vec!["x".to_string(), "z".to_string()])
    );
    assert_eq!(
        payload.training_feature_ranges,
        Some(vec![(0.0, 1.0), (-3.5, 4.0)])
    );
}

#[test]
fn survival_marginal_slope_saved_models_require_special_predict_handling() {
    let mut payload = FittedModelPayload::new(
        "Surv(t0, t1, event) ~ s(x)".to_string(),
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::RoystonParmar,
                InverseLink::Standard(StandardLink::Identity),
            ),
            survival_likelihood: Some("marginal-slope".to_string()),
            survival_distribution: Some(ResidualDistribution::Gaussian),
            frailty: FrailtySpec::None,
        },
        "survival".to_string(),
    );
    payload.fit_result = Some(minimal_survival_fit_result());
    let model = FittedModel::from_payload(payload);

    assert_eq!(model.predict_model_class(), PredictModelClass::Survival);
    assert!(
        model.predictor().is_err(),
        "saved survival marginal-slope models should bypass the generic predictor path"
    );
}
