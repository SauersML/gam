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
use ndarray::Array1;
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

