use gam::families::survival::lognormal_kernel::FrailtySpec;
use gam::inference::model::{
    FittedEstimator, FittedFamily, FittedModel, FittedModelPayload, MODEL_PAYLOAD_VERSION,
    ModelKind,
};
use gam::types::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, LinkComponent, MixtureLinkState,
    ResponseFamily, SasLinkState, StandardLink,
};
use ndarray::array;

#[test]
fn fitted_family_all_variants_round_trip_to_identical_json_bytes() {
    let variants = vec![
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::new(
                ResponseFamily::Binomial,
                InverseLink::Standard(StandardLink::Logit),
            ),
            link: Some(StandardLink::Logit),
            latent_cloglog_state: Some(LatentCLogLogState { latent_sd: 1.25 }),
            mixture_state: Some(MixtureLinkState {
                components: vec![LinkComponent::Probit, LinkComponent::Logit],
                rho: array![0.2],
                pi: array![0.55, 0.45],
            }),
            sas_state: Some(SasLinkState {
                epsilon: 0.1,
                log_delta: -0.2,
                delta: 0.9,
            }),
        },
        FittedFamily::LocationScale {
            likelihood: LikelihoodSpec::gaussian_identity(),
            base_link: Some(InverseLink::Standard(StandardLink::Identity)),
        },
        FittedFamily::MarginalSlope {
            likelihood: LikelihoodSpec::binomial_probit(),
            base_link: InverseLink::Standard(StandardLink::Probit),
            frailty: FrailtySpec::None,
        },
        FittedFamily::Survival {
            likelihood: LikelihoodSpec::royston_parmar(),
            survival_likelihood: Some("aft".to_string()),
            survival_distribution: None,
            frailty: FrailtySpec::None,
        },
        FittedFamily::TransformationNormal {
            likelihood: LikelihoodSpec::gaussian_identity(),
        },
    ];

    for variant in variants {
        let bytes = serde_json::to_vec(&variant)
            .expect("each FittedFamily variant should serialize to JSON bytes");
        let decoded: FittedFamily =
            serde_json::from_slice(&bytes).expect("serialized FittedFamily should deserialize");
        let reencoded =
            serde_json::to_vec(&decoded).expect("deserialized FittedFamily should serialize again");
        assert_eq!(
            bytes, reencoded,
            "each FittedFamily variant should round-trip to the same JSON bytes"
        );
    }
}

#[test]
fn fitted_family_likelihood_returns_variant_specific_likelihood() {
    let ls = LikelihoodSpec::gaussian_identity();
    let ms = LikelihoodSpec::binomial_probit();
    let sv = LikelihoodSpec::royston_parmar();
    let tn = LikelihoodSpec::binomial_logit();

    assert_eq!(
        FittedFamily::Standard {
            likelihood: ls.clone(),
            link: None,
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        }
        .likelihood(),
        ls,
        "Standard variant should return its own likelihood"
    );
    assert_eq!(
        FittedFamily::LocationScale {
            likelihood: LikelihoodSpec::gaussian_identity(),
            base_link: None,
        }
        .likelihood(),
        LikelihoodSpec::gaussian_identity(),
        "LocationScale variant should return its own likelihood"
    );
    assert_eq!(
        FittedFamily::MarginalSlope {
            likelihood: ms.clone(),
            base_link: InverseLink::Standard(StandardLink::Probit),
            frailty: FrailtySpec::None,
        }
        .likelihood(),
        ms,
        "MarginalSlope variant should return its own likelihood"
    );
    assert_eq!(
        FittedFamily::Survival {
            likelihood: sv.clone(),
            survival_likelihood: None,
            survival_distribution: None,
            frailty: FrailtySpec::None,
        }
        .likelihood(),
        sv,
        "Survival variant should return its own likelihood"
    );
    assert_eq!(
        FittedFamily::TransformationNormal {
            likelihood: tn.clone(),
        }
        .likelihood(),
        tn,
        "TransformationNormal variant should return its own likelihood"
    );
}

#[test]
fn standard_family_parameterized_link_states_survive_serde_round_trip() {
    let original = FittedFamily::Standard {
        likelihood: LikelihoodSpec::binomial_logit(),
        link: Some(StandardLink::Logit),
        latent_cloglog_state: Some(LatentCLogLogState { latent_sd: 2.0 }),
        mixture_state: Some(MixtureLinkState {
            components: vec![LinkComponent::Logit, LinkComponent::CLogLog],
            rho: array![-0.35],
            pi: array![0.4, 0.6],
        }),
        sas_state: Some(SasLinkState {
            epsilon: 0.2,
            log_delta: 0.3,
            delta: 1.4,
        }),
    };

    let bytes = serde_json::to_vec(&original).expect("standard family should serialize");
    let decoded: FittedFamily =
        serde_json::from_slice(&bytes).expect("standard family should deserialize");
    assert_eq!(
        serde_json::to_value(original).expect("original family should convert to JSON value"),
        serde_json::to_value(decoded).expect("decoded family should convert to JSON value"),
        "latent_cloglog_state, mixture_state, and sas_state should survive serde round-trip"
    );
}

#[test]
fn payload_with_older_version_is_rejected_with_version_mismatch() {
    let payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION - 1,
        "y ~ 1".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::gaussian_identity(),
            link: Some(StandardLink::Identity),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "gaussian".to_string(),
    );

    let err = FittedModel::from_payload(payload)
        .validate_for_persistence()
        .expect_err("older payload versions should fail with a schema version mismatch");
    assert!(
        err.to_string().contains("MODEL_PAYLOAD_VERSION"),
        "version mismatch errors should explicitly mention MODEL_PAYLOAD_VERSION"
    );
}

#[test]
fn estimator_metadata_is_required_and_expectile_tau_is_validated() {
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "y ~ 1".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: LikelihoodSpec::gaussian_identity(),
            link: Some(StandardLink::Identity),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        "expectile(0.9)".to_string(),
    );
    payload.estimator = FittedEstimator::Expectile { tau: 0.9 };

    let mut encoded = serde_json::to_value(&payload).expect("serialize current payload");
    encoded
        .as_object_mut()
        .expect("payload JSON object")
        .remove("estimator");
    assert!(
        serde_json::from_value::<FittedModelPayload>(encoded).is_err(),
        "v12 must not decode a saved model whose estimator identity is absent"
    );

    payload.estimator = FittedEstimator::Expectile { tau: 1.0 };
    let error = FittedModel::from_payload(payload)
        .validate_for_persistence()
        .expect_err("tau=1 is not an expectile target");
    assert!(error.to_string().contains("strictly in (0, 1)"));
}

#[test]
fn from_payload_model_kind_maps_to_expected_fitted_model_variant() {
    let cases = vec![
        (ModelKind::Standard, "standard"),
        (ModelKind::LocationScale, "location-scale"),
        (ModelKind::MarginalSlope, "marginal-slope"),
        (ModelKind::Survival, "survival"),
        (ModelKind::TransformationNormal, "transformation-normal"),
    ];

    for (kind, label) in cases {
        let family_state = match kind {
            ModelKind::Standard => FittedFamily::Standard {
                likelihood: LikelihoodSpec::gaussian_identity(),
                link: Some(StandardLink::Identity),
                latent_cloglog_state: None,
                mixture_state: None,
                sas_state: None,
            },
            ModelKind::LocationScale => FittedFamily::LocationScale {
                likelihood: LikelihoodSpec::gaussian_identity(),
                base_link: Some(InverseLink::Standard(StandardLink::Identity)),
            },
            ModelKind::MarginalSlope => FittedFamily::MarginalSlope {
                likelihood: LikelihoodSpec::binomial_probit(),
                base_link: InverseLink::Standard(StandardLink::Probit),
                frailty: FrailtySpec::None,
            },
            ModelKind::Survival => FittedFamily::Survival {
                likelihood: LikelihoodSpec::royston_parmar(),
                survival_likelihood: None,
                survival_distribution: None,
                frailty: FrailtySpec::None,
            },
            ModelKind::TransformationNormal => FittedFamily::TransformationNormal {
                likelihood: LikelihoodSpec::gaussian_identity(),
            },
        };

        let payload = FittedModelPayload::new(
            MODEL_PAYLOAD_VERSION,
            "y ~ 1".to_string(),
            kind,
            family_state,
            "family".to_string(),
        );

        let model = FittedModel::from_payload(payload);
        assert_eq!(
            model.model_kind, kind,
            "model kind '{label}' should map to the matching FittedModel variant"
        );
    }
}
