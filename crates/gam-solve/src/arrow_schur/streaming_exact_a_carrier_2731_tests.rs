//! Regression pins for #2731's chunked exact-A evidence contract.

use super::*;
use ndarray::array;

fn refusing_options() -> ArrowSolveOptions {
    ArrowSolveOptions::direct()
        .with_indefinite_refusing_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR)
}

#[test]
fn streaming_chunk_carries_exact_a_operands_from_the_source_system() {
    let mut system = ArrowSchurSystem::new(1, 1, 1);
    system.rows[0].htt[[0, 0]] = 1.0;
    system.hbb[[0, 0]] = 2.0;
    system.exact_a_classification = Some(ExactAClassificationGeometry {
        rows: vec![ExactAClassificationRow {
            delta_tt: array![[0.0]],
            delta_tbeta: array![[0.0]],
            clamp_diag: array![0.0],
        }]
        .into(),
        border_indices: vec![0].into(),
    });

    let mut streaming = StreamingArrowSchur::from_system(&system, 1);
    let contribution = streaming
        .evidence_schur_chunk(0.0, 0.0, &refusing_options())
        .expect("the streaming policy must travel with its exact-A operands");
    assert_eq!(contribution.log_det_tt, 0.0);
    assert_eq!(contribution.schur, array![[2.0]]);
    assert_eq!(contribution.majorizer_metric, Some(array![[2.0]]));
    assert_eq!(contribution.clamp_metric, Some(array![[0.0]]));
}

#[test]
fn streaming_reduced_schur_prices_a_clamp_basin_with_its_carrier() {
    let schur = array![[-0.5]];
    let majorizer = array![[1.0]];
    let clamp = array![[2.0]];
    let value = StreamingArrowSchur::reduced_schur_log_det(
        &schur,
        &refusing_options(),
        Some(&majorizer),
        Some(&clamp),
    )
    .expect("the clamp restores positive basin curvature");
    assert!((value - 1.5_f64.ln()).abs() <= 8.0 * f64::EPSILON);
}

#[test]
fn streaming_refusing_policy_without_exact_a_carrier_still_declines() {
    let error =
        StreamingArrowSchur::reduced_schur_log_det(&array![[1.0]], &refusing_options(), None, None)
            .expect_err("a verdict without its operands must remain an error");
    assert!(error.to_string().contains("raw B/delta/clamp carrier"));
}

#[test]
fn streaming_reduced_schur_rejects_a_partial_carrier() {
    let error = StreamingArrowSchur::reduced_schur_log_det(
        &array![[1.0]],
        &refusing_options(),
        Some(&array![[1.0]]),
        None,
    )
    .expect_err("one metric cannot classify the exact operator");
    assert!(error.to_string().contains("partial exact-A"));
}

#[test]
fn non_refusing_unit_deflation_does_not_require_exact_a_geometry() {
    let options =
        ArrowSolveOptions::direct().with_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR);
    let value = StreamingArrowSchur::reduced_schur_log_det(&array![[-0.5]], &options, None, None)
        .expect("legacy unit deflation has no exact-A classification contract");
    assert_eq!(value, 0.0, "unit stiffness contributes log(1) = 0");
}
