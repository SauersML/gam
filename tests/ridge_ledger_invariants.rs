//! Invariants for `StabilizationLedger`.
//!
//! These tests pin the canonical taxonomy of stabilization ridges:
//!
//!   * `SolverDampingOnly`     — never enters objective/grad/Hessian/logdet.
//!   * `NumericalPerturbation` — never enters objective/grad/Hessian/logdet.
//!   * `ExplicitPrior`         — enters every accounting pass consistently
//!                               (objective up by ½ δ ‖β‖², gradient gains
//!                               δ β, Hessian gains δ I, logdet gains the
//!                               appropriate term).
//!
//! Other agents in the cross-cutting ridge cleanup (penalty_strict,
//! pirls_curvature, covariance_strict, resource_serialize, custom_family)
//! coordinate by reading these invariants, so the asserts here are the
//! stable contract the rest of the codebase keys off.

use gam::types::{
    RidgeMatrixForm, RidgePassport, RidgePolicy, StabilizationKind, StabilizationLedger,
    StabilizationRule,
};
use ndarray::Array1;

const DELTA: f64 = 7.5e-3;

#[test]
fn solver_damping_excludes_every_accounting_term() {
    let ledger = StabilizationLedger::solver_damping(DELTA, StabilizationRule::FixedConstant);
    assert!(ledger.invariants_hold());
    assert_eq!(ledger.delta, DELTA);
    assert!(!ledger.included_in_quadratic);
    assert!(!ledger.included_in_laplace_hessian);
    assert!(!ledger.included_in_penalty_logdet);
    assert_eq!(ledger.quadratic_delta(), 0.0);
    assert_eq!(ledger.laplace_hessian_delta(), 0.0);
    assert_eq!(ledger.penalty_logdet_delta(), 0.0);
    assert!(matches!(ledger.kind, StabilizationKind::SolverDampingOnly));
}

#[test]
fn numerical_perturbation_excludes_every_accounting_term() {
    let ledger = StabilizationLedger::numerical_perturbation(
        DELTA,
        StabilizationRule::InertiaTarget { spd_floor: 1e-10 },
        Some(1e-12),
    );
    assert!(ledger.invariants_hold());
    assert!(!ledger.included_in_quadratic);
    assert!(!ledger.included_in_laplace_hessian);
    assert!(!ledger.included_in_penalty_logdet);
    assert_eq!(ledger.quadratic_delta(), 0.0);
    assert_eq!(ledger.laplace_hessian_delta(), 0.0);
    assert_eq!(ledger.penalty_logdet_delta(), 0.0);
    match ledger.kind {
        StabilizationKind::NumericalPerturbation {
            backward_error_bound,
        } => {
            assert_eq!(backward_error_bound, Some(1e-12));
        }
        _ => panic!("expected NumericalPerturbation"),
    }
}

#[test]
fn explicit_prior_includes_every_accounting_term() {
    let ledger = StabilizationLedger::explicit_prior(DELTA, RidgeMatrixForm::ScaledIdentity);
    assert!(ledger.invariants_hold());
    assert!(ledger.included_in_quadratic);
    assert!(ledger.included_in_laplace_hessian);
    assert!(ledger.included_in_penalty_logdet);
    assert_eq!(ledger.quadratic_delta(), DELTA);
    assert_eq!(ledger.laplace_hessian_delta(), DELTA);
    assert_eq!(ledger.penalty_logdet_delta(), DELTA);
    assert!(matches!(ledger.kind, StabilizationKind::ExplicitPrior));
}

/// Numerical model: confirm that an ExplicitPrior ridge δ moves objective,
/// gradient, and Hessian *consistently* (objective up by ½ δ ‖β‖², gradient
/// up by δ β, Hessian up by δ I), while a SolverDampingOnly ridge with the
/// same δ leaves all of them unchanged. This is the same invariant the
/// PIRLS / survival paths rely on, but exercised against a tiny synthetic
/// model so the invariant is enforced in CI without a full PIRLS run.
#[test]
fn explicit_prior_changes_objective_grad_hessian_consistently() {
    let beta = Array1::from_vec(vec![0.4, -1.1, 2.0]);
    let beta_norm_sq = beta.dot(&beta);

    let prior = StabilizationLedger::explicit_prior(DELTA, RidgeMatrixForm::ScaledIdentity);
    let damping = StabilizationLedger::solver_damping(DELTA, StabilizationRule::FixedConstant);

    // Synthetic baseline objective f0(β), gradient g0, Hessian h0_diag.
    let f0 = 1.234;
    let g0 = Array1::from_vec(vec![0.5, -0.25, 0.1]);
    let h0_diag = Array1::from_vec(vec![1.0, 1.0, 1.0]);

    // ExplicitPrior path: every accounting term shifts by exactly δ-units.
    let f_prior = f0 + 0.5 * prior.quadratic_delta() * beta_norm_sq;
    let g_prior = &g0 + &beta.mapv(|v| prior.quadratic_delta() * v);
    let h_prior_diag = &h0_diag + &Array1::from_elem(3, prior.laplace_hessian_delta());

    assert!((f_prior - (f0 + 0.5 * DELTA * beta_norm_sq)).abs() < 1e-12);
    for k in 0..3 {
        assert!((g_prior[k] - (g0[k] + DELTA * beta[k])).abs() < 1e-12);
        assert!((h_prior_diag[k] - (h0_diag[k] + DELTA)).abs() < 1e-12);
    }

    // SolverDampingOnly path: every accounting term is unchanged. δ is real
    // (the LM step uses it inside a temporary trust-region linear solve), but
    // the *model* the caller observes through the ledger is indistinguishable
    // from the no-stabilization baseline.
    let f_damp = f0 + 0.5 * damping.quadratic_delta() * beta_norm_sq;
    let g_damp = &g0 + &beta.mapv(|v| damping.quadratic_delta() * v);
    let h_damp_diag = &h0_diag + &Array1::from_elem(3, damping.laplace_hessian_delta());

    assert_eq!(f_damp, f0);
    assert_eq!(g_damp, g0);
    assert_eq!(h_damp_diag, h0_diag);
}

/// Bridge from the legacy `RidgePassport` API: a passport whose policy
/// turns on every inclusion flag must round-trip into an
/// `ExplicitPrior`-kind ledger (so PIRLS-side code that already threads a
/// passport through every call can hand the ledger downstream without
/// reclassifying it). A passport with all flags off must round-trip into
/// `NumericalPerturbation`.
#[test]
fn from_passport_round_trip_preserves_classification() {
    let prior_passport =
        RidgePassport::scaled_identity(DELTA, RidgePolicy::explicit_stabilization_full());
    let prior_ledger = StabilizationLedger::from_passport(prior_passport);
    assert!(prior_ledger.invariants_hold());
    assert!(matches!(
        prior_ledger.kind,
        StabilizationKind::ExplicitPrior
    ));
    assert_eq!(prior_ledger.delta, DELTA);

    // A policy with every flag false: morally a numerical perturbation.
    let np_policy = RidgePolicy {
        rho_independent: true,
        include_quadratic_penalty: false,
        include_penalty_logdet: false,
        include_laplacehessian: false,
        determinant_mode: gam::types::RidgeDeterminantMode::Auto,
    };
    let np_passport = RidgePassport::scaled_identity(DELTA, np_policy);
    let np_ledger = StabilizationLedger::from_passport(np_passport);
    assert!(np_ledger.invariants_hold());
    assert!(matches!(
        np_ledger.kind,
        StabilizationKind::NumericalPerturbation { .. }
    ));
}

/// `StabilizationLedger::none()` must satisfy every invariant trivially:
/// δ = 0, all flags off, kind = None. Any future call site that wants a
/// "no-stabilization" sentinel pulls this rather than constructing an
/// inconsistent zero-delta-with-some-flag-on record by hand.
#[test]
fn none_sentinel_holds_invariants() {
    let ledger = StabilizationLedger::none();
    assert!(ledger.invariants_hold());
    assert_eq!(ledger.delta, 0.0);
    assert!(matches!(ledger.kind, StabilizationKind::None));
}
