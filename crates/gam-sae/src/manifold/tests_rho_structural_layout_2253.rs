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
fn invalid_constructor_rho_is_refused_before_bounds_or_fixed_fit_2253() {
    let mut bounded = planted_periodic_outer_objective_2253();
    bounded.baseline_rho.log_lambda_smooth[0] = LOG_STRENGTH_MAX + 1.0;
    let error = bounded
        .outer_domain_lower_bound()
        .expect_err("an invalid constructor-supplied baseline must not be projected into domain");
    assert!(
        error.to_string().contains("smoothness log strength"),
        "unexpected baseline-domain error: {error}"
    );

    let mut fixed = planted_periodic_outer_objective_2253();
    let mut flat = fixed.baseline_rho.flat_coordinates();
    flat[fixed.baseline_rho.ard_flat_index(0, 0)] = LOG_STRENGTH_MIN - 1.0;
    let error = fixed
        .fit_at_fixed_rho(flat.view())
        .expect_err("fixed-rho entry must reject before any inner solve");
    assert!(
        error.to_string().contains("ARD log precision"),
        "unexpected fixed-rho domain error: {error}"
    );
}

/// A one-row assignment of `k` atoms under `mode`: the layout reads only the assignment
/// family and its effective concentration predicate.
fn layout_assignment(mode: AssignmentMode, k: usize) -> SaeAssignment {
    SaeAssignment::from_blocks_with_mode_and_manifolds(
        ndarray::Array2::<f64>::zeros((1, k)),
        vec![ndarray::Array2::<f64>::zeros((1, 1)); k],
        vec![LatentManifold::Euclidean; k],
        mode,
    )
    .expect("one logit column, coordinate block and manifold per atom")
}

#[test]
fn fixed_assignment_strength_is_absent_from_flat_rho_layout_2253() {
    let softmax = SaeManifoldRho::new(-1.7, 0.4, vec![array![-0.2]])
        .for_assignment(&layout_assignment(AssignmentMode::softmax(0.8), 1));
    assert_eq!(softmax.sparse_flat_index(), None);
    assert_eq!(softmax.smooth_flat_index(0), 0);
    assert_eq!(softmax.ard_flat_index(0, 0), 1);
    assert_eq!(softmax.flat_coordinates(), array![0.4, -0.2]);

    // Reconstitution moves only the two mathematical coordinates and retains
    // the stored (inner-state) sparse value without emitting it into the outer
    // vector.
    let moved = array![0.7, -0.5];
    let restored = softmax.from_flat(moved.view()).unwrap();
    assert_eq!(restored.flat_coordinates(), moved);
    assert_abs_diff_eq!(restored.log_lambda_sparse, -1.7, epsilon = 0.0);

    // TopK has no assignment-strength penalty at any K: the fixed support is
    // the sparsity constraint itself.
    let topk = SaeManifoldRho::new(-0.9, 0.1, vec![array![0.2], array![0.3]])
        .for_assignment(&layout_assignment(AssignmentMode::top_k_support(1), 2));
    assert_eq!(topk.sparse_flat_index(), None);
    assert_eq!(topk.flat_coordinates(), array![0.1, 0.1, 0.2, 0.3]);

    // Softmax regains the assignment-strength coordinate automatically when a
    // second atom makes entropy non-constant.
    let two_atom_softmax = SaeManifoldRho::new(-0.9, 0.1, vec![array![0.2], array![0.3]])
        .for_assignment(&layout_assignment(AssignmentMode::softmax(0.8), 2));
    assert_eq!(two_atom_softmax.sparse_flat_index(), Some(0));
    assert_eq!(two_atom_softmax.smooth_flat_index(0), 1);
}

/// #2933 F45 — an ordered Beta--Bernoulli prior carries the sparse coordinate exactly while
/// its concentration is learned. A fixed concentration is the complete prior at weight one,
/// and its placeholder is kept out of the flat vector and left unchanged by reconstitution.
#[test]
fn ordered_beta_bernoulli_sparse_coordinate_follows_the_effective_concentration_2933() {
    let smooth_and_ard = vec![array![0.2], array![0.3]];
    let learnable = layout_assignment(AssignmentMode::ordered_beta_bernoulli(0.8, 1.7, true), 2);
    let learned = SaeManifoldRho::new(-0.9, 0.1, smooth_and_ard.clone()).for_assignment(&learnable);
    assert_eq!(learned.sparse_flat_index(), Some(0));
    assert_eq!(learned.flat_coordinates(), array![-0.9, 0.1, 0.1, 0.2, 0.3]);

    let fixed = layout_assignment(AssignmentMode::ordered_beta_bernoulli(0.8, 1.7, false), 2);
    let rho = SaeManifoldRho::new(-0.9, 0.1, smooth_and_ard.clone()).for_assignment(&fixed);
    assert_eq!(rho.sparse_flat_index(), None);
    assert_eq!(rho.flat_coordinates(), array![0.1, 0.1, 0.2, 0.3]);
    let restored = rho
        .from_flat(array![0.4, -0.2, 0.5, 0.6].view())
        .expect("the four present coordinates rebuild");
    assert_abs_diff_eq!(restored.log_lambda_sparse, -0.9, epsilon = 0.0);
}

