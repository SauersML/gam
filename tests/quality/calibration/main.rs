use gam::test_support::calibration::{
    CalibrationKind, CalibrationRng, CalibrationTarget, FAST_COVERAGE_REPLICATES, FAST_SBC_DRAWS,
    IntervalDraw, SBC_DECILE_BINS, UncertaintyField, audit_interval_coverage,
    audit_registry_completeness, audit_sbc_ranks,
};

const GAUSSIAN_REFERENCE_TARGET: CalibrationTarget = CalibrationTarget::new(
    "GaussianReference.mean_interval",
    CalibrationKind::Interval,
    0x1891_1869_1878,
);
const NARROW_INTERVAL_TARGET: CalibrationTarget = CalibrationTarget::new(
    "PlantedMiscalibration.narrow_interval",
    CalibrationKind::Interval,
    0x1891_1870_1875,
);
const SBC_REFERENCE_TARGET: CalibrationTarget = CalibrationTarget::new(
    "GaussianReference.posterior_rank",
    CalibrationKind::PosteriorSbc,
    0x1891_1841_1810,
);
const SKEWED_SBC_TARGET: CalibrationTarget = CalibrationTarget::new(
    "PlantedMiscalibration.skewed_rank",
    CalibrationKind::PosteriorSbc,
    0x1891_1878_1841,
);

fn gaussian_interval_draws(seed: u64, standard_error_scale: f64) -> Vec<IntervalDraw> {
    let mut rng = CalibrationRng::new(seed);
    (0..FAST_COVERAGE_REPLICATES)
        .map(|_| {
            let error = rng.standard_normal();
            IntervalDraw {
                truth: 0.0,
                estimate: error,
                standard_error: standard_error_scale,
            }
        })
        .collect()
}

fn uniform_sbc_ranks(draws_per_rank: usize) -> Vec<usize> {
    (0..FAST_SBC_DRAWS)
        .map(|draw| draw % (draws_per_rank + 1))
        .collect()
}

#[test]
fn registry_completeness_lint_rejects_unregistered_uncertainty_payload_field() {
    let fields = [
        UncertaintyField::new("PredictiveMean", "se"),
        UncertaintyField::new("PredictiveMean", "interval"),
        UncertaintyField::new("LikelihoodRatio", "p_value"),
    ];
    let incomplete = [CalibrationTarget::new(
        "PredictiveMean.se",
        CalibrationKind::Interval,
        0x1891_0001,
    )];
    let failure = audit_registry_completeness(&incomplete, &fields).expect_err(
        "an uncertainty payload carrying interval/p_value fields must be registered before it ships",
    );
    assert_eq!(
        failure.missing_targets,
        vec![
            "LikelihoodRatio.p_value".to_string(),
            "PredictiveMean.interval".to_string()
        ]
    );

    let complete = [
        CalibrationTarget::new("PredictiveMean.se", CalibrationKind::Interval, 0x1891_0001),
        CalibrationTarget::new(
            "PredictiveMean.interval",
            CalibrationKind::Interval,
            0x1891_0002,
        ),
        CalibrationTarget::new(
            "LikelihoodRatio.p_value",
            CalibrationKind::TestPValue,
            0x1891_0003,
        ),
    ];
    audit_registry_completeness(&complete, &fields)
        .expect("complete calibration registry must pass the lint");
}

#[test]
fn gaussian_reference_interval_passes_and_planted_narrow_interval_fails() {
    let reference = gaussian_interval_draws(GAUSSIAN_REFERENCE_TARGET.seed, 1.0);
    let reference_verdicts = audit_interval_coverage(&GAUSSIAN_REFERENCE_TARGET, &reference);
    assert!(
        reference_verdicts.iter().all(|verdict| verdict.passed),
        "closed-form Gaussian reference should pass nominal coverage: {reference_verdicts:?}"
    );

    let narrowed = gaussian_interval_draws(NARROW_INTERVAL_TARGET.seed, 0.5);
    let narrowed_verdicts = audit_interval_coverage(&NARROW_INTERVAL_TARGET, &narrowed);
    assert!(
        narrowed_verdicts.iter().any(|verdict| !verdict.passed),
        "a deliberately halved interval must fail anti-conservative coverage: {narrowed_verdicts:?}"
    );
    assert!(
        narrowed_verdicts
            .iter()
            .filter(|verdict| !verdict.passed)
            .all(|verdict| verdict.empirical < verdict.nominal),
        "planted failure must be under-coverage, not conservative noise: {narrowed_verdicts:?}"
    );
}

#[test]
fn sbc_reference_rank_histogram_passes_and_planted_skew_fails() {
    let draws_per_rank = SBC_DECILE_BINS - 1;
    let reference_ranks = uniform_sbc_ranks(draws_per_rank);
    let reference = audit_sbc_ranks(
        &SBC_REFERENCE_TARGET,
        &reference_ranks,
        draws_per_rank,
        SBC_DECILE_BINS,
    );
    assert!(
        reference.passed,
        "uniform planted SBC ranks should pass the rank-histogram gate: {reference:?}"
    );

    let skewed_ranks = vec![0usize; FAST_SBC_DRAWS];
    let skewed = audit_sbc_ranks(
        &SKEWED_SBC_TARGET,
        &skewed_ranks,
        draws_per_rank,
        SBC_DECILE_BINS,
    );
    assert!(
        !skewed.passed,
        "a deliberately one-sided posterior rank distribution must fail SBC uniformity: {skewed:?}"
    );
}
