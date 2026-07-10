//! E3 primitive integration (gam#2234): a two-layer crosscoder must decode the
//! downstream layer at the same steered chart coordinate as the anchor.
//!
//! This fixture installs one exact period-1 circle atom with two decoder blocks.
//! The anchor block decodes `q(t) = (sin 2πt, cos 2πt)` and the downstream block
//! decodes `q(t + φ)`.  There is only one stored coordinate per row.  After a
//! chart step `δ`, the two honest-layer accessors must therefore produce
//! `q(t + δ)` in the anchor block and `q(t + δ + φ)` in the downstream block.
//! No downstream coordinate solve or second fit is involved.

use super::*;
use ndarray::{Array2, array};
use std::sync::Arc;

fn circular_pair(turns: f64) -> [f64; 2] {
    let angle = std::f64::consts::TAU * turns;
    [angle.sin(), angle.cos()]
}

#[test]
fn e3_crosscoder_handoff_uses_one_shared_moved_coordinate() {
    let n = 48usize;
    let phase = 0.173_205_080_756_887_73;
    let delta = 0.1375;
    let block_log_lambda = 1.4;
    let block_sqrt_lambda = (0.5_f64 * block_log_lambda).exp();
    let evaluator = Arc::new(PeriodicHarmonicEvaluator::new(3).unwrap());
    let coords = Array2::<f64>::from_shape_fn((n, 1), |(row, _)| row as f64 / n as f64);
    let (phi, jet) = evaluator.evaluate(coords.view()).unwrap();

    // PeriodicHarmonicEvaluator(3) emits [1, sin(2πt), cos(2πt)].  The leading
    // two output columns are the anchor and the trailing two are layer 1.
    let mut decoder = Array2::<f64>::zeros((3, 4));
    decoder[[1, 0]] = 1.0;
    decoder[[2, 1]] = 1.0;
    let phase_angle = std::f64::consts::TAU * phase;
    // A production crosscoder stores downstream columns in the weighted fit
    // space, √λ·B.  The E3 handoff must remove this non-unit √λ before returning
    // an activation to patch into the downstream residual stream.
    decoder[[1, 2]] = block_sqrt_lambda * phase_angle.cos();
    decoder[[1, 3]] = -block_sqrt_lambda * phase_angle.sin();
    decoder[[2, 2]] = block_sqrt_lambda * phase_angle.sin();
    decoder[[2, 3]] = block_sqrt_lambda * phase_angle.cos();

    let atom = SaeManifoldAtom::new(
        "e3-shared-circle",
        SaeAtomBasisKind::Periodic,
        1,
        phi,
        jet,
        decoder,
        Array2::<f64>::eye(3),
    )
    .unwrap()
    .with_basis_evaluator(evaluator);
    let assignment = SaeAssignment::from_blocks_with_mode_and_manifolds(
        Array2::<f64>::zeros((n, 1)),
        vec![coords.clone()],
        vec![LatentManifold::Circle { period: 1.0 }],
        AssignmentMode::softmax(1.0),
    )
    .unwrap();
    let mut term = SaeManifoldTerm::new(vec![atom], assignment).unwrap();
    term.set_crosscoder_layout(
        CrosscoderLayout::new(
            2,
            vec![2],
            vec!["downstream".into()],
            vec![block_log_lambda],
        )
        .unwrap(),
    )
    .unwrap();

    let rows: Vec<usize> = (0..n).collect();
    let moved_anchor = term
        .steer_layer_decode(0, CrosscoderLayer::Anchor, &rows, array![delta].view())
        .expect("shared-coordinate anchor steer");
    let moved_downstream = term
        .steer_layer_decode(0, CrosscoderLayer::Block(0), &rows, array![delta].view())
        .expect("shared-coordinate downstream steer");
    let downstream_delta = term
        .steer_layer_delta(0, CrosscoderLayer::Block(0), &rows, array![delta].view())
        .expect("shared-coordinate downstream delta");
    let base_downstream = term
        .steer_layer_decode(0, CrosscoderLayer::Block(0), &rows, array![0.0].view())
        .expect("shared-coordinate downstream baseline");

    assert_eq!(moved_anchor.dim(), (n, 2));
    assert_eq!(moved_downstream.dim(), (n, 2));
    assert_eq!(term.assignment.coords.len(), 1, "the two decoder blocks share one chart");
    for row in 0..n {
        let t = coords[[row, 0]];
        let expected_anchor = circular_pair(t + delta);
        let expected_downstream = circular_pair(t + delta + phase);
        for col in 0..2 {
            assert!(
                (moved_anchor[[row, col]] - expected_anchor[col]).abs() < 1.0e-12,
                "anchor row {row}, col {col}: got {}, expected {}",
                moved_anchor[[row, col]],
                expected_anchor[col]
            );
            assert!(
                (moved_downstream[[row, col]] - expected_downstream[col]).abs() < 1.0e-12,
                "downstream row {row}, col {col}: got {}, expected {}",
                moved_downstream[[row, col]],
                expected_downstream[col]
            );
            assert!(
                (downstream_delta[[row, col]]
                    - (moved_downstream[[row, col]] - base_downstream[[row, col]]))
                    .abs()
                    < 1.0e-12,
                "downstream delta must be the moved-minus-base contribution"
            );
        }
    }
}
