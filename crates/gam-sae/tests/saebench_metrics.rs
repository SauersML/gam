use gam_sae::saebench_metrics::{
    ChartInterpObservation, DoseResponseObservation, chart_interp_score,
    coordinate_posterior_from_precision, dose_response_calibration,
};

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
    let report = chart_interp_score(&obs).unwrap();
    assert!(
        report.circular_correlation > 0.99,
        "orientation-quotiented cyclic chart score should recover the reversed coordinate: {report:?}"
    );
    assert!(report.signed_circular_correlation < 0.0);
}

#[test]
fn dose_response_reports_fisher_calibration_slope_and_unit_speed_constancy() {
    let obs = [
        DoseResponseObservation {
            arc_length: 1.0,
            predicted_nats: 0.5,
            measured_nats: 1.0,
            weight: 1.0,
        },
        DoseResponseObservation {
            arc_length: 2.0,
            predicted_nats: 1.0,
            measured_nats: 2.0,
            weight: 1.0,
        },
        DoseResponseObservation {
            arc_length: 3.0,
            predicted_nats: 1.5,
            measured_nats: 3.0,
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
