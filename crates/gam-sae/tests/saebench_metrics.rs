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
    let report = chart_interp_score(&obs, &[null_draw], 0.05).unwrap();
    assert!(
        report.circular_correlation > 0.99,
        "orientation-quotiented cyclic chart score should recover the reversed coordinate: {report:?}"
    );
    assert!(report.signed_circular_correlation < 0.0);
    assert!(
        !report.evidentially_valid,
        "one null draw cannot attain p <= 0.05"
    );
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

fn perfect_null_draw() -> Vec<ChartInterpObservation> {
    correlation_fixture(1.0)
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

fn calibrated_fixture(
    target: f64,
    exceedances: usize,
) -> gam_sae::saebench_metrics::ChartInterpReport {
    let mut draws = Vec::with_capacity(2_499);
    draws.extend((0..exceedances).map(|_| perfect_null_draw()));
    draws.extend((exceedances..2_499).map(|_| zero_null_draw()));
    chart_interp_score(&correlation_fixture(target), &draws, 0.05).unwrap()
}

#[test]
fn chart_interp_calibration_fixtures_pin_weekday_month_and_color_verdicts_2250() {
    // 399 exceedances among 2,499 draws => (399 + 1) / 2,500 = 0.16.
    let weekday = calibrated_fixture(0.973, 399);
    assert!((weekday.circular_correlation - 0.973).abs() < 1.0e-12);
    assert!((weekday.matched_spectrum_p_value - 0.16).abs() < 1.0e-12);
    assert!(!weekday.evidentially_valid);

    // Zero exceedances => the plus-one Monte Carlo floor 1 / 2,500 = 0.0004.
    let month = calibrated_fixture(0.95, 0);
    assert!((month.matched_spectrum_p_value - 0.0004).abs() < 1.0e-12);
    assert!(month.evidentially_valid);

    // 2,324 exceedances => (2,324 + 1) / 2,500 = 0.93.
    let color = calibrated_fixture(0.125, 2_324);
    assert!((color.matched_spectrum_p_value - 0.93).abs() < 1.0e-12);
    assert!(!color.evidentially_valid);
}

#[test]
fn chart_interp_rejects_scalar_only_evidence_2250() {
    let error = chart_interp_score(&correlation_fixture(0.973), &[], 0.05).unwrap_err();
    assert!(error.contains("matched-spectrum null draw"));
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
