use super::*;
use ndarray::{Array1, array};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn run_fixture(continuation_is_stationary: bool) -> Result<OuterResult, EstimationError> {
    let incumbent = array![7.25];

    let continuation_requested = Arc::new(AtomicBool::new(false));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(incumbent.clone())
        .with_max_iter(5);
    let mut objective = problem.build_objective(
        (),
        {
            let continuation_requested = Arc::clone(&continuation_requested);
            let incumbent = incumbent.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                if !continuation_requested.load(Ordering::Relaxed) {
                    return Ok(100.0 - theta[0]);
                }
                if theta == &incumbent {
                    return Ok(4.0);
                }
                Err(EstimationError::TrialPointRefused {
                    reason: "synthetic continuation neighbourhood is outside the domain"
                        .to_string(),
                })
            }
        },
        {
            let continuation_requested = Arc::clone(&continuation_requested);
            let incumbent = incumbent.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                if theta == &incumbent {
                    return Ok(OuterEval {
                        cost: 4.0,
                        gradient: array![if continuation_is_stationary { 0.0 } else { 1.0 }],
                        hessian: HessianValue::Unavailable,
                        inner_beta_hint: None,
                    });
                }
                if !continuation_requested.load(Ordering::Relaxed) {
                    return Ok(OuterEval {
                        cost: 100.0 - theta[0],
                        gradient: array![-1.0],
                        hessian: HessianValue::Unavailable,
                        inner_beta_hint: None,
                    });
                }
                Err(EstimationError::TrialPointRefused {
                    reason: "synthetic continuation neighbourhood is outside the domain"
                        .to_string(),
                })
            }
        },
        None::<fn(&mut ())>,
        {
            let continuation_requested = Arc::clone(&continuation_requested);
            let incumbent = incumbent.clone();
            Some(move |_: &mut (), theta: &Array1<f64>| {
                if theta == &incumbent {
                    return Ok(EfsEval {
                        cost: 100.0 - theta[0],
                        steps: vec![0.25],
                        beta: None,
                        psi_gradient: None,
                        psi_indices: None,
                        inner_hessian_scale: None,
                        consecutive_restored_incumbents: None,
                    });
                }
                continuation_requested.store(true, Ordering::Relaxed);
                Err(EstimationError::TrialPointRefused {
                    reason: "synthetic EFS trial refusal".to_string(),
                })
            })
        },
    );

    problem.run(&mut objective, "fixed-point continuation #2653")
}

/// #2653: a refused EFS trial hands the run to the BFGS continuation from the
/// fixed-point incumbent. A stationary incumbent certifies there; a
/// nonstationary one whose whole neighbourhood is outside the domain is
/// refused. No seed lattice stands behind the continuation to substitute an
/// unrelated point for the incumbent the fixed point actually reached.
#[test]
fn fixed_point_continuation_certifies_only_a_stationary_incumbent_2653() {
    let result = run_fixture(true).expect("a stationary continuation incumbent certifies");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    assert_eq!(result.rho, array![7.25]);

    run_fixture(false).expect_err(
        "a nonstationary incumbent with a refused neighbourhood has no certified optimum",
    );
}
