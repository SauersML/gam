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
    let restored = softmax.from_flat(moved.view()).unwrap();
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
    let mut flat = fixed.baseline_rho.to_flat();
    flat[fixed.baseline_rho.ard_flat_index(0, 0)] = LOG_STRENGTH_MIN - 1.0;
    let error = fixed
        .fit_at_fixed_rho(flat.view())
        .expect_err("fixed-rho entry must reject before any inner solve");
    assert!(
        error.to_string().contains("ARD log precision"),
        "unexpected fixed-rho domain error: {error}"
    );
}

/// Per-coordinate, per-channel FIXED-STATE criterion-derivative audit — the
/// single-source instrument for the outer criterion (#2253, stage 3.3.iii).
///
/// For every ρ coordinate: central FD of the FROZEN criterion (the
/// `inner_max_iter == 0` verbatim-reuse contract holds θ̂ bit-for-bit fixed,
/// so the FD sees ONLY the direct ρ-dependence) versus the fixed-state
/// analytic channel sum `explicit + logdet_trace + occam` (the adjoint /
/// third-order channel is the θ̂-response and is correctly EXCLUDED at fixed
/// state). A mismatch therefore isolates a genuine channel-formula desync —
/// and the failure message prints each channel separately so the desync NAMES
/// its channel (the instrument that localizes the historical ARD fixed-state
/// anomaly, FD +0.208 vs analytic +1.066). Trap-immunity: no re-solve is on
/// the FD path, so an under-converged inner state cannot fake either side.
#[test]
fn frozen_state_per_coordinate_channel_fd_audit_2253() {
    let mut objective = planted_periodic_outer_objective_2253();
    let base = objective.baseline_rho.to_flat();
    objective
        .eval_cost(&base)
        .expect("base value lane must converge");
    objective
        .eval(&base)
        .expect("base gradient lane must converge");
    let rho_state = objective.baseline_rho.from_flat(base.view()).unwrap();
    let mut audit_term = objective.term.clone();
    let (_frozen_value, audit_loss, audit_cache) = audit_term
        .penalized_quasi_laplace_criterion_with_cache(
            objective.target.view(),
            &rho_state,
            objective.registry.as_ref(),
            0,
            objective.learning_rate,
            objective.ridge_ext_coord,
            objective.ridge_beta,
        )
        .expect("frozen accepted-state evidence audit must evaluate");
    let audit_solver = audit_term
        .outer_gradient_arrow_solver(&audit_cache, &rho_state.lambda_smooth_vec().unwrap())
        .expect("frozen accepted-state outer solver");
    let components = audit_term
        .analytic_outer_rho_gradient_components(
            objective.target.view(),
            &rho_state,
            &audit_loss,
            &audit_cache,
            &audit_solver,
        )
        .expect("frozen accepted-state gradient components");

    let anchor_term = objective.term.clone();
    let frozen_cost_at = |rho_flat: &Array1<f64>| -> f64 {
        let mut term = anchor_term.clone();
        let rho = objective.baseline_rho.from_flat(rho_flat.view()).unwrap();
        term.penalized_quasi_laplace_criterion_with_cache(
            objective.target.view(),
            &rho,
            objective.registry.as_ref(),
            0,
            objective.learning_rate,
            objective.ridge_ext_coord,
            objective.ridge_beta,
        )
        .expect("frozen per-coordinate value probe")
        .0
    };

    let h = 1.0e-4_f64;
    for idx in 0..base.len() {
        let mut plus = base.clone();
        plus[idx] += h;
        let mut minus = base.clone();
        minus[idx] -= h;
        let fd = (frozen_cost_at(&plus) - frozen_cost_at(&minus)) / (2.0 * h);
        let analytic =
            components.explicit[idx] + components.logdet_trace[idx] + components.occam[idx];
        let scale = analytic.abs().max(fd.abs()).max(1.0e-6);
        let rel = (analytic - fd).abs() / scale;
        assert!(
            rel <= 1.0e-3,
            "fixed-state channel desync at rho[{idx}]: analytic={analytic:.9e} \
             (explicit={:.9e}, logdet_trace={:.9e}, occam={:.9e}, \
             excluded third_order={:.9e}) vs frozen central FD={fd:.9e} (rel={rel:.3e}). \
             The desync lives in whichever channel above disagrees with the FD split.",
            components.explicit[idx],
            components.logdet_trace[idx],
            components.occam[idx],
            components.third_order_correction[idx],
        );
    }
}
