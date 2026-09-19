use super::*;

#[test]
fn caller_boolean_and_zero_gradient_cannot_mint_outer_authority() {
    let mut fabricated = OuterResult::new(
        Array1::from_vec(vec![0.0]),
        1.0,
        3,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    fabricated.final_grad_norm = Some(0.0);
    fabricated.final_measurement = Some(OuterFirstOrderMeasurement::new(
        Array1::from_vec(vec![0.0]),
        1.0,
        Array1::from_vec(vec![0.0]),
    ));
    fabricated.termination = OuterTermination::Certified(OuterConvergedVia::GradientStationary);

    let reason = CertifiedOuterResult::from_optimizer_result(fabricated)
        .expect_err("caller-written status and gradient must not mint a certificate");
    assert!(reason.contains("no analytic certificate"), "{reason}");
}
