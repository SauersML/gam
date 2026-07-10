use gam_sae::null_battery::{NullKind, Tail};
use gam_sae::saebench_metrics::{
    ChartInterpNullCalibration, ChartInterpNullDrawPolicy, ChartInterpNullProtocol,
    ChartInterpObservation, ChartInterpStatistic, ChartInterpVerdict, DoseResponseObservation,
    chart_interp_score, coordinate_posterior_from_precision, dose_response_calibration,
};

fn matched_spectrum_calibration(
    seed: u64,
    draws: Vec<Vec<ChartInterpObservation>>,
) -> ChartInterpNullCalibration {
    ChartInterpNullCalibration::new(
        ChartInterpNullProtocol::MatchedSpectrumGaussianChartRefitV1,
        seed,
        draws.len(),
        draws,
    )
    .expect("complete matched-spectrum calibration")
}

#[test]
fn chart_interp_wraps_cyclic_boundary_and_quotients_orientation() {
    let obs = [
        ChartInterpObservation {
            recovered_turns: 0.99,
            label_turns: 0.01,
            weight: 1.0,
        },
        ChartInterpObservation {
            recovered_turns: 0.24,
            label_turns: 0.76,
            weight: 1.0,
        },
        ChartInterpObservation {
            recovered_turns: 0.49,
            label_turns: 0.51,
            weight: 1.0,
        },
        ChartInterpObservation {
            recovered_turns: 0.74,
            label_turns: 0.26,
            weight: 1.0,
        },
    ];
    let null_draw = vec![
        ChartInterpObservation {
            recovered_turns: 0.0,
            label_turns: 0.0,
            weight: 1.0,
        },
        ChartInterpObservation {
            recovered_turns: 0.0,
            label_turns: 0.25,
            weight: 1.0,
        },
        ChartInterpObservation {
            recovered_turns: 0.0,
            label_turns: 0.5,
            weight: 1.0,
        },
        ChartInterpObservation {
            recovered_turns: 0.0,
            label_turns: 0.75,
            weight: 1.0,
        },
    ];
    let calibration = matched_spectrum_calibration(7, vec![null_draw]);
    let report = chart_interp_score(&obs, &calibration, 0.05).unwrap();
    assert!(
        report.observed.circular_correlation > 0.99,
        "orientation-quotiented cyclic chart score should recover the reversed coordinate: {report:?}"
    );
    assert!(report.observed.signed_circular_correlation < 0.0);
    assert_eq!(report.verdict, ChartInterpVerdict::NullCompatible);
}

fn correlation_fixture(target: f64) -> Vec<ChartInterpObservation> {
    let phase_error = target.acos() / std::f64::consts::TAU;
    (0..12)
        .map(|index| {
            let label = index as f64 / 12.0;
            let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            ChartInterpObservation {
                recovered_turns: (label + sign * phase_error).rem_euclid(1.0),
                label_turns: label,
                weight: 1.0,
            }
        })
        .collect()
}

fn zero_null_draw() -> Vec<ChartInterpObservation> {
    (0..12)
        .map(|index| ChartInterpObservation {
            recovered_turns: 0.0,
            label_turns: index as f64 / 12.0,
            weight: 1.0,
        })
        .collect()
}

#[test]
fn chart_interp_report_carries_typed_provenance_and_recomputed_samples_2250() {
    let draws = vec![
        correlation_fixture(0.9),
        zero_null_draw(),
        correlation_fixture(0.6),
    ];
    let calibration = matched_spectrum_calibration(0x2250, draws);
    let report = chart_interp_score(&correlation_fixture(0.8), &calibration, 0.05).unwrap();

    assert_eq!(
        report.statistic,
        ChartInterpStatistic::OrientationQuotientedWeightedPhaseLock
    );
    assert_eq!(report.calibration.statistic, report.statistic);
    assert_eq!(
        report.calibration.protocol,
        ChartInterpNullProtocol::MatchedSpectrumGaussianChartRefitV1
    );
    assert_eq!(
        report.calibration.draw_policy,
        ChartInterpNullDrawPolicy::RegenerateRefitAndReadout
    );
    assert_eq!(
        report.calibration.null_kind,
        NullKind::MatchedSpectrumGaussian
    );
    assert_eq!(report.calibration.seed, 0x2250);
    let null = &report.calibration.null_distribution;
    assert_eq!(null.kind, NullKind::MatchedSpectrumGaussian);
    assert_eq!(null.tail, Tail::Larger);
    assert_eq!(null.n, 3);
    assert_eq!(null.extreme_draws, 1);
    assert_eq!(null.samples.len(), 3);
    assert!((null.samples[0] - 0.9).abs() < 1.0e-12);
    assert!(null.samples[1].abs() < 1.0e-12);
    assert!((null.samples[2] - 0.6).abs() < 1.0e-12);
    assert!((null.p_value - 0.5).abs() < 1.0e-12);
    assert_eq!(report.verdict, ChartInterpVerdict::NullCompatible);
}

#[test]
fn chart_interp_rejects_scalar_only_evidence_2250() {
    let error = ChartInterpNullCalibration::new(
        ChartInterpNullProtocol::MatchedSpectrumGaussianChartRefitV1,
        0x2250,
        0,
        Vec::new(),
    )
    .unwrap_err();
    assert!(error.contains("at least one draw"));

    let error = ChartInterpNullCalibration::new(
        ChartInterpNullProtocol::MatchedSpectrumGaussianChartRefitV1,
        0x2250,
        2,
        vec![zero_null_draw()],
    )
    .unwrap_err();
    assert!(error.contains("declares 2 draws"));
}

#[test]
fn chart_interp_rejects_a_null_ledger_for_different_labels_2250() {
    let observed = correlation_fixture(0.8);
    let mut wrong_labels = zero_null_draw();
    wrong_labels[0].label_turns = 0.25;
    let calibration = matched_spectrum_calibration(0x2250, vec![wrong_labels]);
    let error = chart_interp_score(&observed, &calibration, 0.05).unwrap_err();
    assert!(error.contains("changes label_turns"));
}

#[test]
fn dose_response_reports_fisher_calibration_slope_and_unit_speed_constancy() {
    // Endpoint dosing is quadratic in the applied chord: predicted = ½·arc²,
    // measured = arc², so the per-arc² rate is exactly constant (CV = 0) and
    // the calibration slope through the origin is exactly 2.
    let obs = [
        DoseResponseObservation {
            arc_length: 1.0,
            predicted_nats: 0.5,
            measured_nats: 1.0,
            weight: 1.0,
        },
        DoseResponseObservation {
            arc_length: 2.0,
            predicted_nats: 2.0,
            measured_nats: 4.0,
            weight: 1.0,
        },
        DoseResponseObservation {
            arc_length: 3.0,
            predicted_nats: 4.5,
            measured_nats: 9.0,
            weight: 1.0,
        },
    ];
    let report = dose_response_calibration(&obs).unwrap();
    assert!((report.slope_through_origin - 2.0).abs() < f64::EPSILON.sqrt());
    assert!((report.r2_through_origin - 1.0).abs() < f64::EPSILON.sqrt());
    assert!(report.cv_measured_nats_per_arc_squared < f64::EPSILON.sqrt());
}

#[test]
fn coordinate_posterior_inverts_row_hessian_precision_block() {
    let posterior =
        coordinate_posterior_from_precision(&[0.25, 0.75], &[4.0, 1.0, 1.0, 3.0]).unwrap();
    assert!((posterior.covariance_diag[0] - 3.0 / 11.0).abs() < f64::EPSILON.sqrt());
    assert!((posterior.covariance_diag[1] - 4.0 / 11.0).abs() < f64::EPSILON.sqrt());
    assert!((posterior.precision_weight - 11.0 / 7.0).abs() < f64::EPSILON.sqrt());
}
