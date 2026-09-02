//! #2253/#2658 — the EFS / HybridEFS first-order fallback is a typed routing
//! request, and it has to be honoured wherever it is raised.
//!
//! `OuterFixedPointBridge` emits [`FirstOrderFallbackRequest`] when the
//! fixed-point step is not a descent direction it can rescue (ψ stagnation, or
//! a step every halving rejected on both the full vector and the ρ/τ-only
//! fallback). `automatic_fallback_attempts` builds exactly the plan it is
//! asking for — the `disable_fixed_point` BFGS attempt — for any
//! analytic-gradient EFS/HybridEFS primary.
//!
//! The seed evaluation in `run_fixed_point_outer_solver` routes the request to
//! `ImmediateFallback`. Every LATER iteration used to surface instead as
//! `FixedPointError::ObjectiveFailed`, which was classified
//! `fatal_outer_evaluation` after its typed source was discarded — and a fatal
//! classification short-circuits both `run_outer_with_plan`'s request
//! propagation and the attempt loop in `run_outer_with_strategy`. So a search
//! that DESCENDED and then asked to hand over to the joint gradient solver died
//! with the fallback plan never attempted. Measured on the #2253 planted-circle
//! fixture (n=48, p=6, K=2, circle, softmax): HybridEFS descends 5.992e1 →
//! 2.414e1 and then returns
//! `Fatal outer-objective evaluation failure (outer fixed-point evaluation):
//!  … HybridEFS step rejected after 8 halvings on full vector and 8
//!  halvings on ρ/τ-only fallback`.

use super::*;
use crate::inner_status::InnerFailure;
use gam_problem::{CustomFamilyError, InnerConvergenceTerminalState};

/// The seed EFS evaluation succeeds and starts the fixed-point walk; a LATER
/// iteration raises the typed fallback request. The run must degrade to the
/// BFGS plan the request asks for and converge, not fail the whole fit.
#[test]
fn typed_efs_fallback_raised_after_the_seed_degrades_to_bfgs_2253_2658() {
    let efs_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(3)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(Array1::from_elem(3, 1.0))
        .with_max_iter(20);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.5 * theta.dot(theta),
                gradient: theta.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        {
            let efs_calls = Arc::clone(&efs_calls);
            Some(move |_: &mut (), theta: &Array1<f64>| {
                let call = efs_calls.fetch_add(1, Ordering::Relaxed);
                if call == 0 {
                    // Seed validation: a real descending step, so the
                    // fixed-point walk actually starts and the request below is
                    // raised from inside `FixedPoint::run`, not from the seed
                    // screen the old routing already handled.
                    Ok(EfsEval {
                        cost: 0.5 * theta.dot(theta),
                        steps: vec![-0.25_f64; theta.len()],
                        beta: None,
                        psi_gradient: None,
                        psi_indices: None,
                        inner_hessian_scale: None,
                        logdet_enclosure_gap: None,
                        consecutive_restored_incumbents: None,
                    })
                } else {
                    Err(EstimationError::GradientUnavailable {
                        context: "synthetic post-seed EFS step rejection",
                        mode: "typed first-order fallback request",
                    })
                }
            })
        },
    );

    let result = problem
        .run(&mut obj, "post-seed EFS fallback request")
        .expect("a post-seed first-order fallback request must degrade to BFGS, not fail the fit");
    assert_eq!(
        result.plan_used.solver,
        Solver::Bfgs,
        "the request asks for the joint gradient solver; the run must have used it"
    );
    assert!(
        result.converged(),
        "the degraded BFGS plan solves this quadratic and must certify"
    );
    assert!(
        efs_calls.load(Ordering::Relaxed) >= 2,
        "the request must be raised AFTER the seed evaluation (calls={}), \
         otherwise this test re-covers the seed-time routing instead",
        efs_calls.load(Ordering::Relaxed)
    );
}

/// The other post-seed route in gam#2658: a rho-local custom-family refusal
/// must cross `opt::FixedPoint` with its complete producer source, not return as
/// a fatal string. Startup accounting consumes the exact same error after this
/// adapter, so checking both layers here seals the lossy boundary itself.
#[test]
fn post_seed_custom_family_refusal_retains_typed_terminal_state_2658() {
    let terminal = InnerConvergenceTerminalState::JointNewton {
        cycle: 11,
        stationarity_residual: 2.5e-3,
        residual_tol: 1.0e-8,
        // Consistent with the tol above: `tol = 1e-11 · (1 + scale)`.
        stationarity_scale: 999.0,
        step_inf: 4.0e-4,
        step_tol: 1.0e-9,
        resolvable_negative_curvature: false,
        best_stationarity_residual: 7.5e-5,
        cycles_since_best_residual: 3,
        termination_reason: gam_problem::JointNewtonTerminalReason::CycleBudget,
    };
    let efs_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(3)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_max_iter(20);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.5 * theta.dot(theta),
                gradient: theta.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        {
            let efs_calls = Arc::clone(&efs_calls);
            Some(move |_: &mut (), theta: &Array1<f64>| {
                if efs_calls.fetch_add(1, Ordering::Relaxed) == 0 {
                    return Ok(EfsEval {
                        cost: 0.5 * theta.dot(theta),
                        steps: vec![-0.25_f64; theta.len()],
                        beta: None,
                        psi_gradient: None,
                        psi_indices: None,
                        inner_hessian_scale: None,
                        logdet_enclosure_gap: None,
                        consecutive_restored_incumbents: None,
                    });
                }
                Err(EstimationError::CustomFamily(
                    CustomFamilyError::InnerSolveNotConverged {
                        cycles: 12,
                        terminal: Some(terminal),
                        kkt_residual: Some(2.5e-3),
                        kkt_tol: Some(1.0e-8),
                        theta_dim: 3,
                        rho_dim: 3,
                        psi_dim: 0,
                    },
                ))
            })
        },
    );
    let capability = obj.capability();
    let the_plan = plan(&capability);
    assert_eq!(the_plan.solver, Solver::Efs);
    let seed = Array1::from_elem(3, 1.0);
    let failure = match run_fixed_point_outer_solver(
        &mut obj,
        capability.theta_layout(),
        capability.barrier_config.clone(),
        &problem.config(),
        "typed post-seed custom-family refusal",
        &seed,
        the_plan,
        "EFS",
        "EFS failed",
    ) {
        Err(failure) => failure,
        Ok(_) => panic!("the synthetic second EFS evaluation must refuse"),
    };
    let request = match failure {
        FixedPointOuterRunError::IterationRejected(request) => request,
        FixedPointOuterRunError::SeedRejected(_) => {
            panic!("the first EFS evaluation succeeded; this is not a seed rejection")
        }
        FixedPointOuterRunError::ImmediateFallback(_) => {
            panic!("custom-family non-convergence is a rho-local refusal, not a solver request")
        }
        FixedPointOuterRunError::Failed(error) => {
            panic!("rho-local custom-family refusal was made fatal: {error}")
        }
    };
    let objective_error = &request.refusal;
    assert!(objective_error.is_recoverable());
    let source = objective_error
        .downcast_ref::<EstimationError>()
        .expect("the EstimationError source must cross opt::FixedPoint");
    assert!(matches!(
        source,
        EstimationError::CustomFamily(CustomFamilyError::InnerSolveNotConverged {
            cycles: 12,
            terminal: Some(observed_terminal),
            kkt_residual: Some(2.5e-3),
            kkt_tol: Some(1.0e-8),
            theta_dim: 3,
            rho_dim: 3,
            psi_dim: 0,
        }) if *observed_terminal == terminal
    ));
    let rejection = SeedRejection::from_objective_error(0, "solver", objective_error);
    assert!(matches!(
        rejection.failure,
        InnerFailure::InnerSolveNotConverged {
            source: CustomFamilyError::InnerSolveNotConverged {
                cycles: 12,
                terminal: Some(observed_terminal),
                ..
            },
            ..
        } if observed_terminal == terminal
    ));
}

/// A non-finite criterion at the literal seed is evidence about that seed, not
/// a structural failure of the objective. The fixed-point adapter must retain
/// the bridge's recoverable verdict so the caller can continue the seed
/// cascade.
#[test]
fn non_finite_efs_seed_cost_is_a_typed_seed_rejection_2653() {
    let problem = OuterProblem::new(3)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.5 * theta.dot(theta),
                gradient: theta.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        Some(|_: &mut (), theta: &Array1<f64>| {
            Ok(EfsEval {
                cost: f64::INFINITY,
                steps: vec![0.0; theta.len()],
                beta: None,
                psi_gradient: None,
                psi_indices: None,
                inner_hessian_scale: None,
                logdet_enclosure_gap: None,
                consecutive_restored_incumbents: None,
            })
        }),
    );
    let capability = obj.capability();
    let the_plan = plan(&capability);
    assert_eq!(the_plan.solver, Solver::Efs);
    let seed = Array1::from_elem(3, 1.0);

    let error = match run_fixed_point_outer_solver(
        &mut obj,
        capability.theta_layout(),
        capability.barrier_config.clone(),
        &problem.config(),
        "non-finite EFS seed cost",
        &seed,
        the_plan,
        "EFS",
        "EFS failed",
    ) {
        Err(FixedPointOuterRunError::SeedRejected(error)) => error,
        Err(FixedPointOuterRunError::IterationRejected(_)) => {
            panic!("the first EFS evaluation failed; no iteration started")
        }
        Err(FixedPointOuterRunError::ImmediateFallback(_)) => {
            panic!("a non-finite cost is a point-local refusal, not a solver request")
        }
        Err(FixedPointOuterRunError::Failed(error)) => {
            panic!("a point-local non-finite seed cost was made fatal: {error}")
        }
        Ok(_) => panic!("a non-finite EFS seed cost must be rejected"),
    };
    assert!(error.is_recoverable());
    assert_eq!(
        error.message(),
        "outer EFS eval failed: objective returned a non-finite cost"
    );
}

/// The same producer verdict after a successful seed evaluation must remain an
/// iteration rejection. This is the boundary that used to discard the typed
/// error and abort all remaining seeds.
#[test]
fn non_finite_post_seed_efs_cost_is_a_typed_iteration_rejection_2653() {
    let efs_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(3)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_max_iter(20);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.5 * theta.dot(theta),
                gradient: theta.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        {
            let efs_calls = Arc::clone(&efs_calls);
            Some(move |_: &mut (), theta: &Array1<f64>| {
                let cost = if efs_calls.fetch_add(1, Ordering::Relaxed) == 0 {
                    0.5 * theta.dot(theta)
                } else {
                    f64::INFINITY
                };
                Ok(EfsEval {
                    cost,
                    steps: vec![-0.25; theta.len()],
                    beta: Some(Array1::from_elem(theta.len(), 42.0)),
                    psi_gradient: None,
                    psi_indices: None,
                    inner_hessian_scale: None,
                    logdet_enclosure_gap: None,
                    consecutive_restored_incumbents: None,
                })
            })
        },
    );
    let capability = obj.capability();
    let the_plan = plan(&capability);
    assert_eq!(the_plan.solver, Solver::Efs);
    let seed = Array1::from_elem(3, 1.0);

    let request = match run_fixed_point_outer_solver(
        &mut obj,
        capability.theta_layout(),
        capability.barrier_config.clone(),
        &problem.config(),
        "non-finite post-seed EFS cost",
        &seed,
        the_plan,
        "EFS",
        "EFS failed",
    ) {
        Err(FixedPointOuterRunError::IterationRejected(request)) => request,
        Err(FixedPointOuterRunError::SeedRejected(_)) => {
            panic!("the finite seed evaluation succeeded; this is not a seed rejection")
        }
        Err(FixedPointOuterRunError::ImmediateFallback(_)) => {
            panic!("a non-finite cost is a point-local refusal, not a solver request")
        }
        Err(FixedPointOuterRunError::Failed(error)) => {
            panic!("a point-local non-finite iteration cost was made fatal: {error}")
        }
        Ok(_) => panic!("the synthetic post-seed non-finite cost must be rejected"),
    };
    let error = &request.refusal;
    assert!(error.is_recoverable());
    assert_eq!(
        error.message(),
        "outer EFS eval failed: objective returned a non-finite cost"
    );
    assert_eq!(
        efs_calls.load(Ordering::Relaxed),
        2,
        "one finite seed evaluation and one rejected iteration must be observed"
    );
    assert_eq!(request.checkpoint.point, seed);
    assert_eq!(
        request.checkpoint.sample.value.to_bits(),
        (1.5_f64).to_bits()
    );
    assert_eq!(request.checkpoint.sample.step, Array1::from_elem(3, -0.25));
    assert_eq!(request.checkpoint.iterations, 0);
    assert_eq!(request.checkpoint.plan_used.solver, Solver::Efs);
    let inner_seed = request
        .checkpoint
        .inner_seed
        .expect("the continuation must preserve beta from the exact finite rho");
    assert_eq!(inner_seed.theta, seed);
    assert_eq!(inner_seed.beta, Array1::from_elem(3, 42.0));
}

/// #2653 root cause: a rho-local refusal belongs to the proposed next point,
/// not to the finite EFS incumbent that proposed it.  The automatic fallback
/// must start BFGS exactly once at that incumbent; generating another EFS or
/// BFGS seed destroys the basin evidence the completed iterations already
/// established.
#[test]
fn rho_local_efs_refusal_resumes_bfgs_from_last_finite_incumbent_once_2653() {
    let efs_calls = Arc::new(AtomicUsize::new(0));
    let refusal_seen = Arc::new(AtomicBool::new(false));
    let first_fallback_eval = Arc::new(Mutex::new(None::<Array1<f64>>));
    let initial = Array1::from_elem(3, 2.0);
    let expected_checkpoint = Array1::from_elem(3, 1.75);
    let problem = OuterProblem::new(3)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(initial)
        .with_max_iter(40);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
        {
            let refusal_seen = Arc::clone(&refusal_seen);
            let first_fallback_eval = Arc::clone(&first_fallback_eval);
            move |_: &mut (), theta: &Array1<f64>| {
                if refusal_seen.load(Ordering::Relaxed) {
                    let mut slot = first_fallback_eval
                        .lock()
                        .expect("fallback-point observation lock poisoned");
                    if slot.is_none() {
                        *slot = Some(theta.clone());
                    }
                }
                Ok(OuterEval {
                    cost: 0.5 * theta.dot(theta),
                    gradient: theta.clone(),
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        {
            let efs_calls = Arc::clone(&efs_calls);
            let refusal_seen = Arc::clone(&refusal_seen);
            Some(move |_: &mut (), theta: &Array1<f64>| {
                let call = efs_calls.fetch_add(1, Ordering::Relaxed);
                let (cost, step, beta) = match call {
                    0 => (
                        0.5 * theta.dot(theta),
                        vec![-0.25; theta.len()],
                        Some(Array1::from_elem(theta.len(), 7.0)),
                    ),
                    1 => (
                        0.5 * theta.dot(theta),
                        vec![-0.125; theta.len()],
                        Some(Array1::from_elem(theta.len(), 9.0)),
                    ),
                    _ => {
                        refusal_seen.store(true, Ordering::Relaxed);
                        (f64::INFINITY, vec![0.0; theta.len()], None)
                    }
                };
                Ok(EfsEval {
                    cost,
                    steps: step,
                    beta,
                    psi_gradient: None,
                    psi_indices: None,
                    inner_hessian_scale: None,
                    logdet_enclosure_gap: None,
                    consecutive_restored_incumbents: None,
                })
            })
        },
    );

    let result = problem
        .run(&mut obj, "rho-local EFS continuation #2653")
        .expect("BFGS must continue and certify the finite EFS incumbent");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    assert!(result.converged());
    assert_eq!(
        efs_calls.load(Ordering::Relaxed),
        3,
        "the refused EFS proposal must route immediately; no new EFS seed may start"
    );
    assert_eq!(
        first_fallback_eval
            .lock()
            .expect("fallback-point observation lock poisoned")
            .as_ref(),
        Some(&expected_checkpoint),
        "the first fallback evaluation must be the exact last finite EFS point"
    );
}

