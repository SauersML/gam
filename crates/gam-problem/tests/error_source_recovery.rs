//! #2818 recovery of typed optimizer error propagation through the public API.

use gam_problem::CustomFamilyError;
use gam_problem::estimation_error::{EstimationError, OuterObjectiveErrorSource};

#[test]
fn fatal_optimizer_evaluation_retains_exact_typed_source_2658() {
    let original = EstimationError::CustomFamily(CustomFamilyError::InnerSolveNotConverged {
        cycles: 17,
        terminal: None,
        kkt_residual: Some(3.5),
        kkt_tol: Some(0.25),
        theta_dim: 4,
        rho_dim: 3,
        psi_dim: 1,
    });
    let boundary = EstimationError::fatal_objective_evaluation(
        "outer fixed-point evaluation",
        opt::ObjectiveEvalError::fatal_from(original),
    );
    assert!(boundary.is_fatal_outer_evaluation());
    let error = EstimationError::fatal_outer_evaluation("outer orchestration", boundary);
    let EstimationError::OuterObjectiveEvaluationFailed { context, source } = &error else {
        panic!("the terminal error must retain its outer-evaluation boundary type");
    };
    assert_eq!(context, "outer fixed-point evaluation");
    let OuterObjectiveErrorSource::Objective(producer) = source else {
        panic!("the optimizer's source and fatal verdict must remain directly accessible");
    };
    assert!(producer.is_fatal());
    let Some(EstimationError::CustomFamily(CustomFamilyError::InnerSolveNotConverged {
        cycles,
        terminal,
        kkt_residual,
        kkt_tol,
        theta_dim,
        rho_dim,
        psi_dim,
    })) = source.estimation_error()
    else {
        panic!("the typed inner-solve evidence was flattened or replaced");
    };
    assert!(terminal.is_none());
    assert_eq!(
        (
            *cycles,
            *kkt_residual,
            *kkt_tol,
            *theta_dim,
            *rho_dim,
            *psi_dim
        ),
        (17, Some(3.5), Some(0.25), 4, 3, 1),
    );
}
