use gam::test_support::calibration::{
    CalibrationKind, CalibrationTarget, CoverageObservation, SbcRankObservation,
    TestSizeObservation, UncertaintySurface, assert_registry_complete, audit_target,
};

#[test]
fn closed_form_gaussian_reference_passes_coverage_gate() {
    let mut target =
        CalibrationTarget::new("gaussian_mean_wald_interval", CalibrationKind::Coverage);
    for &nominal in &[0.80, 0.90, 0.95] {
        for i in 0..200 {
            let miss_period = match nominal {
                x if (x - 0.80_f64).abs() < 1e-12 => 5,
                x if (x - 0.90_f64).abs() < 1e-12 => 10,
                _ => 20,
            };
            target.coverage.push(CoverageObservation {
                nominal,
                covered: i % miss_period != 0,
            });
        }
    }
    let verdict = audit_target(&target);
    assert!(verdict.passed, "{verdict:?}");
}

#[test]
fn deliberately_narrowed_interval_fails_coverage_gate() {
    let mut target =
        CalibrationTarget::new("planted_half_width_interval", CalibrationKind::Coverage);
    for &nominal in &[0.80, 0.90, 0.95] {
        for i in 0..200 {
            target.coverage.push(CoverageObservation {
                nominal,
                covered: i % 2 == 0,
            });
        }
    }
    let verdict = audit_target(&target);
    assert!(
        !verdict.passed,
        "narrow intervals must be anti-conservative: {verdict:?}"
    );
}

#[test]
fn deliberately_skewed_posterior_fails_sbc_uniformity_gate() {
    let mut target =
        CalibrationTarget::new("planted_skewed_posterior", CalibrationKind::SbcRanks);
    for _ in 0..120 {
        target.sbc.push(SbcRankObservation {
            rank: 0,
            draws: 20,
        });
    }
    let verdict = audit_target(&target);
    assert!(!verdict.passed, "skewed ranks must fail SBC: {verdict:?}");
}

#[test]
fn uniform_sbc_reference_passes() {
    let mut target =
        CalibrationTarget::new("planted_uniform_posterior", CalibrationKind::SbcRanks);
    for i in 0..210 {
        target.sbc.push(SbcRankObservation {
            rank: i % 21,
            draws: 20,
        });
    }
    let verdict = audit_target(&target);
    assert!(verdict.passed, "{verdict:?}");
}

#[test]
fn anti_conservative_test_size_fails() {
    let mut target = CalibrationTarget::new("planted_lr_p_value", CalibrationKind::TestSize);
    for &alpha in &[0.01, 0.05, 0.10] {
        for i in 0..200 {
            target.test_size.push(TestSizeObservation {
                alpha,
                rejected: i % 5 == 0,
            });
        }
    }
    let verdict = audit_target(&target);
    assert!(
        !verdict.passed,
        "anti-conservative null size must fail: {verdict:?}"
    );
}

#[test]
fn registry_completeness_lint_requires_every_uncertainty_surface() {
    let discovered = [
        UncertaintySurface::new("mean_prediction_interval", "PredictionPayload", "interval"),
        UncertaintySurface::new("mean_prediction_se", "PredictionPayload", "se"),
    ];
    let registry = [CalibrationTarget::new(
        "mean_prediction_interval",
        CalibrationKind::Coverage,
    )];
    let err = assert_registry_complete(&discovered, &registry).unwrap_err();
    assert_eq!(err.missing[0].name, "mean_prediction_se");
}
