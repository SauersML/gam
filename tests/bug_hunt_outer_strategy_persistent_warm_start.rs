use gam::solver::outer_strategy::{
    DeclaredHessianForm, Derivative, HessianSource, OuterCapability, Solver, plan,
};

fn cap(
    gradient: Derivative,
    hessian: DeclaredHessianForm,
    n_params: usize,
    psi_dim: usize,
    fixed_point_available: bool,
) -> OuterCapability {
    OuterCapability {
        gradient,
        hessian,
        n_params,
        psi_dim,
        fixed_point_available,
        barrier_config: None,
        prefer_gradient_only: false,
        disable_fixed_point: false,
    }
}

#[test]
fn outer_strategy_selector_prefers_hybrid_efs_for_large_mixed_rho_psi_problem() {
    let c = cap(
        Derivative::Analytic,
        DeclaredHessianForm::Unavailable,
        12,
        3,
        true,
    );
    let p = plan(&c);
    assert_eq!(
        p.solver,
        Solver::HybridEfs,
        "Expected HybridEfs for large mixed rho+psi problems with fixed-point support; selector chose a different inner-strategy variant"
    );
    assert_eq!(
        p.hessian_source,
        HessianSource::HybridEfsFixedPoint,
        "Expected HybridEfsFixedPoint Hessian source for mixed rho+psi fixed-point route"
    );
}

#[test]
fn outer_strategy_selector_prefers_efs_for_large_penalty_only_problem() {
    let c = cap(
        Derivative::Analytic,
        DeclaredHessianForm::Unavailable,
        12,
        0,
        true,
    );
    let p = plan(&c);
    assert_eq!(
        p.solver,
        Solver::Efs,
        "Expected Efs for large penalty-only problems with fixed-point support; selector chose a different inner-strategy variant"
    );
}

#[test]
fn outer_strategy_selector_falls_back_to_bfgs_when_fixed_point_disabled() {
    let mut c = cap(
        Derivative::Analytic,
        DeclaredHessianForm::Unavailable,
        12,
        0,
        true,
    );
    c.disable_fixed_point = true;
    let p = plan(&c);
    assert_eq!(
        p.solver,
        Solver::Bfgs,
        "Expected Bfgs when fixed-point is disabled even with large penalty-only problem"
    );
}

#[test]
fn outer_strategy_selector_cost_only_fails_loudly_with_bfgs() {
    // A cost-only capability (no gradient, no Hessian, no fixed-point lane) is
    // a programming error now that every outer objective supplies an analytic
    // gradient. The planner emits Bfgs — which the runner rejects loudly for
    // needing a gradient the objective cannot supply.
    let c = cap(
        Derivative::Unavailable,
        DeclaredHessianForm::Unavailable,
        3,
        0,
        false,
    );
    let p = plan(&c);
    assert_eq!(
        p.solver,
        Solver::Bfgs,
        "cost-only capability must fail loudly via Bfgs, never a non-gradient solver"
    );
}
