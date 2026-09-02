use super::*;
use ndarray::{Array1, array};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

fn run_fixture(continuation_is_stationary: bool) -> (OuterResult, usize) {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 2;
    let recovery_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config)
        .expect("the bounded recovery lattice must be constructible")
        .into_iter()
        .next()
        .expect("the bounded recovery lattice must contain a seed");
    let incumbent = array![7.25];
    assert_ne!(incumbent, recovery_seed);

    let continuation_requested = Arc::new(AtomicBool::new(false));
    let recovery_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_initial_rho(incumbent.clone())
        .with_max_iter(5);
    let mut objective = problem.build_objective(
        (),
        {
            let continuation_requested = Arc::clone(&continuation_requested);
            let recovery_calls = Arc::clone(&recovery_calls);
            let incumbent = incumbent.clone();
            let recovery_seed = recovery_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                if !continuation_requested.load(Ordering::Relaxed) {
                    return Ok(100.0 - theta[0]);
                }
                if theta == &incumbent {
                    return Ok(4.0);
                }
                if theta == &recovery_seed {
                    recovery_calls.fetch_add(1, Ordering::Relaxed);
                    return Ok(0.0);
                }
                Err(EstimationError::TrialPointRefused {
                    reason: "synthetic continuation neighbourhood is outside the domain"
                        .to_string(),
                })
            }
        },
        {
            let continuation_requested = Arc::clone(&continuation_requested);
            let recovery_calls = Arc::clone(&recovery_calls);
            let incumbent = incumbent.clone();
            let recovery_seed = recovery_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                if theta == &incumbent {
                    return Ok(OuterEval {
                        cost: 4.0,
                        gradient: array![if continuation_is_stationary { 0.0 } else { 1.0 }],
                        hessian: HessianValue::Unavailable,
                        inner_beta_hint: None,
                    });
                }
                if theta == &recovery_seed {
                    recovery_calls.fetch_add(1, Ordering::Relaxed);
                    return Ok(OuterEval {
                        cost: 0.0,
                        gradient: array![0.0],
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
                        logdet_enclosure_gap: None,
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

    let result = problem
        .run(
            &mut objective,
            "fixed-point continuation recovery lattice #2653",
        )
        .expect("the BFGS continuation or its bounded recovery lattice must certify");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    if continuation_is_stationary {
        assert_eq!(result.rho, incumbent);
    } else {
        assert_eq!(result.rho, recovery_seed);
    }
    (result, recovery_calls.load(Ordering::Relaxed))
}

#[test]
fn fixed_point_continuation_uses_recovery_lattice_only_after_incumbent_refusal_2653() {
    let (_, successful_incumbent_recovery_calls) = run_fixture(true);
    assert_eq!(
        successful_incumbent_recovery_calls, 0,
        "a certified continuation incumbent must not launch recovery-seed work"
    );

    let (_, refused_incumbent_recovery_calls) = run_fixture(false);
    assert!(
        refused_incumbent_recovery_calls > 0,
        "a nonstationary continuation incumbent must leave the bounded recovery lattice reachable"
    );
}

