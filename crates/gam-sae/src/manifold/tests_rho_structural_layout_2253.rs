//! #2253 structural rho-layout regressions.
//!
//! A one-atom Softmax assignment is the singleton simplex: every row's gate is
//! exactly one, its entropy is exactly zero, and `log_lambda_sparse` is absent
//! from the mathematical objective. Hard TopK likewise has no sparsity penalty;
//! its fixed support is the sparsity mechanism. These are typed layout absences,
//! not optimizer-held coordinates.

use super::tests::{
    PlantedCircleAssignmentMode, planted_circle_embedded, planted_circle_seed_term,
};
use super::*;
use approx::assert_abs_diff_eq;
use ndarray::{Array1, array};
use std::sync::Arc;

/// Deterministic K=1 periodic objective whose fitted dictionary carries real
/// non-constant signal. The four-row, one-output warm-start contract toy used
/// here previously is intentionally tiny and strongly regularized; its fitted
/// output lands at the column mean and correctly triggers the structural
/// fit-data-collapse ledger. It is therefore a poor active-basin derivative
/// witness; this fixture stays on a well-resolved noncollapsed branch.
fn planted_periodic_outer_objective_2253() -> SaeManifoldOuterObjective {
    let target = planted_circle_embedded(32, 4, 0.02);
    let mut term = planted_circle_seed_term(target.view(), PlantedCircleAssignmentMode::Softmax).0;
    // `planted_circle_seed_term` installs the harmonic evaluator for basis
    // refresh. The analytic logdet-state adjoint also needs its second-jet view.
    term.atoms[0].basis_second_jet = Some(Arc::new(
        PeriodicHarmonicEvaluator::new(3).expect("periodic evaluator"),
    ));
    let rho = SaeManifoldRho::new(0.0, 0.05_f64.ln(), vec![Array1::<f64>::zeros(1)]);
    SaeManifoldOuterObjective::new(term, target, None, rho, 40, 1.0, 1.0e-6, 1.0e-6)
}

#[test]
fn fixed_assignment_strength_is_absent_from_flat_rho_layout_2253() {
    let softmax = SaeManifoldRho::new(-1.7, 0.4, vec![array![-0.2]])
        .for_assignment(AssignmentMode::softmax(0.8));
    assert_eq!(
        softmax.assignment_strength_layout(),
        AssignmentStrengthLayout::SoftmaxEntropy
    );
    assert_eq!(softmax.sparse_flat_index(), None);
    assert_eq!(softmax.smooth_flat_index(0), 0);
    assert_eq!(softmax.ard_flat_index(0, 0), 1);
    assert_eq!(softmax.to_flat(), array![0.4, -0.2]);

    // Reconstitution moves only the two mathematical coordinates and retains
    // the stored (inner-state) sparse value without emitting it into the outer
    // vector.
    let moved = array![0.7, -0.5];
    let restored = softmax.from_flat(moved.view());
    assert_eq!(restored.to_flat(), moved);
    assert_abs_diff_eq!(restored.log_lambda_sparse, -1.7, epsilon = 0.0);

    // TopK has no assignment-strength penalty at any K: the fixed support is
    // the sparsity constraint itself.
    let topk = SaeManifoldRho::new(-0.9, 0.1, vec![array![0.2], array![0.3]])
        .for_assignment(AssignmentMode::top_k_support(1));
    assert_eq!(
        topk.assignment_strength_layout(),
        AssignmentStrengthLayout::FixedSupport
    );
    assert_eq!(topk.sparse_flat_index(), None);
    assert_eq!(topk.to_flat(), array![0.1, 0.1, 0.2, 0.3]);

    // Softmax regains the assignment-strength coordinate automatically when a
    // second atom makes entropy non-constant.
    let two_atom_softmax = SaeManifoldRho::new(-0.9, 0.1, vec![array![0.2], array![0.3]])
        .for_assignment(AssignmentMode::softmax(0.8));
    assert_eq!(two_atom_softmax.sparse_flat_index(), Some(0));
    assert_eq!(two_atom_softmax.smooth_flat_index(0), 1);
}

#[test]
fn k1_softmax_active_rho_gradient_matches_directional_fd_2253() {
    let mut gradient_objective = planted_periodic_outer_objective_2253();
    let base = gradient_objective.baseline_rho.to_flat();
    assert_eq!(
        base.len(),
        2,
        "K=1 Softmax outer rho must contain only smoothness and ARD"
    );
    assert_eq!(gradient_objective.baseline_rho.sparse_flat_index(), None);

    // Match the real optimizer's Value -> ValueAndGradient handoff order.
    let base_cost = gradient_objective
        .eval_cost(&base)
        .expect("base value lane must converge");
    assert!(
        base_cost.is_finite(),
        "the active-rho derivative witness must start at feasible REML evidence: \
         base={base_cost:.17e}"
    );
    let evaluation = gradient_objective
        .eval(&base)
        .expect("base analytic-gradient lane must converge");
    assert!(
        evaluation.cost.is_finite(),
        "the analytic-gradient lane must price the same feasible REML basin: \
         cost={:.17e}",
        evaluation.cost
    );
    assert_eq!(evaluation.gradient.len(), 2);

    let direction = array![0.6_f64, -0.8_f64];
    let analytic = evaluation.gradient.dot(&direction);
    let h = 1.0e-3_f64;
    let plus = &base + &(h * &direction);
    let minus = &base - &(h * &direction);
    // Probe both sides from the accepted base basin, exactly as the production
    // line search does. A fresh cold objective at each side differentiates its
    // seed-dependent basin-entry map rather than the branch whose analytic
    // derivative `evaluation.gradient` describes.
    let plus_cost = gradient_objective
        .eval_cost(&plus)
        .expect("+h directional value probe must converge");
    let plus_telemetry = gradient_objective.probe_telemetry();
    let minus_cost = gradient_objective
        .eval_cost(&minus)
        .expect("-h directional value probe must converge");
    let minus_telemetry = gradient_objective.probe_telemetry();
    assert!(
        plus_cost.is_finite(),
        "+h must evaluate feasible REML evidence: \
         cost={plus_cost:.17e}, telemetry={plus_telemetry:?}"
    );
    assert!(
        minus_cost.is_finite(),
        "-h must evaluate feasible REML evidence: \
         cost={minus_cost:.17e}, telemetry={minus_telemetry:?}"
    );
    assert_eq!(
        plus_telemetry.infeasible_criterion_evals, 0,
        "+h value probe returned infeasible evidence: {plus_telemetry:?}"
    );
    assert_eq!(
        minus_telemetry.infeasible_criterion_evals, 0,
        "-h value probe returned infeasible evidence: {minus_telemetry:?}"
    );
    assert_ne!(
        plus_cost.to_bits(),
        minus_cost.to_bits(),
        "the active-rho value lane is bit-flat across a nonzero directional step: \
         plus={plus_cost:.17e}, minus={minus_cost:.17e}, \
         plus_telemetry={plus_telemetry:?}, minus_telemetry={minus_telemetry:?}"
    );
    let finite_difference = (plus_cost - minus_cost) / (2.0 * h);
    let mut coordinate_fd = Array1::<f64>::zeros(base.len());
    for coordinate in 0..base.len() {
        let mut axis = Array1::<f64>::zeros(base.len());
        axis[coordinate] = 1.0;
        let axis_plus = &base + &(h * &axis);
        let axis_minus = &base - &(h * &axis);
        let axis_plus_cost = gradient_objective
            .eval_cost(&axis_plus)
            .expect("coordinate +h value probe must converge");
        let axis_minus_cost = gradient_objective
            .eval_cost(&axis_minus)
            .expect("coordinate -h value probe must converge");
        coordinate_fd[coordinate] = (axis_plus_cost - axis_minus_cost) / (2.0 * h);
    }
    let scale = analytic.abs().max(finite_difference.abs()).max(1.0);
    assert!(
        (analytic - finite_difference).abs() <= 5.0e-3 * scale,
        "K=1 Softmax active-coordinate derivative mismatch: analytic direction={analytic:.9e}, \
         central FD={finite_difference:.9e}, error={:.3e}, scale={scale:.3e}, \
         gradient={:?}, coordinate_fd={coordinate_fd:?}, \
         base_cost={base_cost:.17e}, plus_cost={plus_cost:.17e}, \
         minus_cost={minus_cost:.17e}",
        (analytic - finite_difference).abs(),
        evaluation.gradient
    );
}
