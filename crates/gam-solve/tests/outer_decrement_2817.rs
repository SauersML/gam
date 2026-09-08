use gam_solve::estimate::EstimationError;
use gam_solve::rho_optimizer::{
    DeclaredHessianForm, Derivative, EfsEval, HessianValue, OuterCriterionCertificate,
    OuterEval, OuterProblem, SeedOutcome,
};
use ndarray::{Array1, array};
use opt::{HessianMaterialization, HessianOperator, ObjectiveEvalError};
use std::sync::Arc;

struct IdentityOperator;

impl HessianOperator for IdentityOperator {
    fn dim(&self) -> usize {
        1
    }

    fn apply_into(
        &self,
        direction: &Array1<f64>,
        output: &mut Array1<f64>,
    ) -> Result<(), ObjectiveEvalError> {
        output.assign(direction);
        Ok(())
    }
}

/// Exercise the public outer driver, including its terminal certificate and
/// retry policy. A small but resolvable gradient can have an unresolved Newton
/// decrement; a caller demanding that gradient still gets the tighter solve.
#[test]
fn matrix_free_decrement_respects_criterion_and_caller_2817() {
    for (required_gradient, install_seed_hook) in [
        (None, false),
        (None, true),
        (Some(1e-10), false),
        (Some(1e-10), true),
    ] {
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Operator {
                materialization: HessianMaterialization::Unavailable,
                estimated_materialization_cost: None,
            })
            .with_initial_rho(array![0.01])
            .with_tolerance(1e-12)
            .with_required_projected_gradient_norm(required_gradient)
            .with_max_iter(5)
            .with_seed_config(gam_problem::SeedConfig {
                max_seeds: 1,
                seed_budget: 1,
                ..Default::default()
            });
        let mut objective = problem
            .build_objective(
                (),
                |_: &mut (), rho: &Array1<f64>| Ok(0.5 * rho.dot(rho)),
                |_: &mut (), rho: &Array1<f64>| {
                    Ok(OuterEval {
                        cost: 0.5 * rho.dot(rho),
                        gradient: rho.clone(),
                        hessian: HessianValue::Operator(Arc::new(IdentityOperator)),
                        inner_beta_hint: None,
                    })
                },
                None::<fn(&mut ())>,
                None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
            )
            .with_criterion_resolution(|_: &mut ()| Some(1e-4));
        // The standard REML builders install the criterion publisher before
        // the coefficient-seed hook. Rebuilding the closure type must retain
        // its numerical contract, including on a run with no cached seed.
        let result = if install_seed_hook {
            let mut seeded = objective.with_seed_inner_state(
                |_: &mut (), _: &Array1<f64>| Ok(SeedOutcome::NoSlot),
            );
            problem.run(&mut seeded, "seeded matrix-free decrement #2817")
        } else {
            problem.run(&mut objective, "matrix-free decrement #2817")
        }
        .expect("the declared criterion and caller requirement must both certify");
        assert!(result.criterion_certificate.as_ref().is_some_and(
            OuterCriterionCertificate::certifies
        ));
        assert!(result.iterations <= 1, "a quadratic needs no retry sweep");
        if required_gradient.is_some() {
            assert!(result.rho[0].abs() <= 1e-10);
        } else {
            assert_eq!(result.rho, array![0.01], "the unresolved step stays untaken");
        }
    }
}
