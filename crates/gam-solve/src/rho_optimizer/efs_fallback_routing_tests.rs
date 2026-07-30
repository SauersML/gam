//! #2253 — the EFS / HybridEFS first-order fallback marker is a ROUTING
//! request, and it has to be honoured wherever it is raised.
//!
//! `OuterFixedPointBridge` emits [`EFS_FIRST_ORDER_FALLBACK_MARKER`] when the
//! fixed-point step is not a descent direction it can rescue (ψ stagnation, or
//! a step every halving rejected on both the full vector and the ρ/τ-only
//! fallback). `automatic_fallback_attempts` builds exactly the plan it is
//! asking for — the `disable_fixed_point` BFGS attempt — for any
//! analytic-gradient EFS/HybridEFS primary.
//!
//! The seed evaluation in `run_fixed_point_outer_solver` has always routed the
//! marker to `ImmediateFallback`. Every LATER iteration surfaced instead as
//! `FixedPointError::ObjectiveFailed`, which was classified
//! `fatal_outer_evaluation` regardless of the marker — and a fatal
//! classification short-circuits both `run_outer_with_plan`'s marker check and
//! the attempt loop in `run_outer_with_strategy`. So a search that DESCENDED
//! and then asked to hand over to the joint gradient solver died with the
//! fallback plan never attempted. Measured on the #2253 planted-circle
//! fixture (n=48, p=6, K=2, circle, softmax): HybridEFS descends 5.992e1 →
//! 2.414e1 and then returns
//! `Fatal outer-objective evaluation failure (outer fixed-point evaluation):
//!  … [outer-efs-first-order-fallback] HybridEFS step rejected after 8
//!  halvings on full vector and 8 halvings on ρ/τ-only fallback`.

use super::*;

/// The seed EFS evaluation succeeds and starts the fixed-point walk; a LATER
/// iteration raises the fallback marker. The run must degrade to the BFGS
/// plan the marker asks for and converge, not fail the whole fit.
#[test]
fn efs_fallback_marker_raised_after_the_seed_degrades_to_bfgs_2253() {
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
                    // fixed-point walk actually starts and the marker below is
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
                    Err(EstimationError::RemlOptimizationFailed(format!(
                        "{EFS_FIRST_ORDER_FALLBACK_MARKER} synthetic post-seed EFS step rejection"
                    )))
                }
            })
        },
    );

    let result = problem
        .run(&mut obj, "post-seed efs fallback marker")
        .expect("a post-seed first-order fallback request must degrade to BFGS, not fail the fit");
    assert_eq!(
        result.plan_used.solver,
        Solver::Bfgs,
        "the marker asks for the joint gradient solver; the run must have used it"
    );
    assert!(
        result.converged(),
        "the degraded BFGS plan solves this quadratic and must certify"
    );
    assert!(
        efs_calls.load(Ordering::Relaxed) >= 2,
        "the marker must be raised AFTER the seed evaluation (calls={}), \
         otherwise this test re-covers the seed-time routing instead",
        efs_calls.load(Ordering::Relaxed)
    );
}

/// A post-seed objective failure that does NOT carry the marker keeps its
/// fatal classification: only the routing request is rerouted.
#[test]
fn post_seed_objective_failure_without_the_marker_stays_fatal_2253() {
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
                    Err(EstimationError::InvalidInput(
                        "synthetic structural EFS defect with no routing request".to_string(),
                    ))
                }
            })
        },
    );

    let error = problem
        .run(&mut obj, "post-seed structural efs failure")
        .expect_err("a structural post-seed failure must not be silently degraded");
    assert!(
        error.is_fatal_outer_evaluation(),
        "an unmarked objective failure keeps its fatal classification, got: {error}"
    );
}

/// The THIRD branch, and the one `ClassifyingFixedPointObjective` was built
/// for: a post-seed failure that carries no routing marker and is NOT
/// structural, but which the producer itself classified `Recoverable`.
///
/// The two tests above pin the ends of the range — a marker is rerouted, a
/// structural `InvalidInput` stays fatal — and between them sits the case the
/// classification machinery exists to serve. `run.rs`'s doc for
/// `ClassifyingFixedPointObjective` names it first: "`Recoverable` for a
/// refusal that is a property of THIS rho (**a non-finite cost**, a non-finite
/// EFS step, a bubbled `RemlOptimizationFailed`, …)", and records that losing
/// that verdict in transit meant "one recoverable per-rho refusal at outer
/// iteration k killed the remaining seeds and the entire fallback ladder".
///
/// Nothing pinned it. A regression that stopped writing or reading the verdict
/// slot for the unmarked-recoverable path would leave BOTH tests above green,
/// because one drives the marker and the other drives a fatal input — so the
/// demotion would silently stop happening on the only branch that needs it.
///
/// A non-finite EFS cost is the cheapest way in: the bridge's own `eval_step`
/// raises exactly `outer EFS eval failed: objective returned a non-finite
/// cost` as `ObjectiveEvalError::recoverable`, which is the same string 19 of
/// the Python suite's failures terminate on (gam#2653).
#[test]
fn post_seed_recoverable_non_finite_cost_is_demoted_not_fatal_2653() {
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
                    // Same descending seed the two tests above use, so the
                    // fixed-point walk genuinely starts and the refusal below
                    // is an ITERATION, not the seed screen.
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
                    // A property of THIS rho, not a broken artifact: the
                    // evaluation succeeded and produced a non-finite value.
                    // The bridge turns this into the recoverable non-finite
                    // cost error; the outer search should step away from this
                    // rho, not abandon the fit.
                    Ok(EfsEval {
                        cost: f64::NAN,
                        steps: vec![-0.25_f64; theta.len()],
                        beta: None,
                        psi_gradient: None,
                        psi_indices: None,
                        inner_hessian_scale: None,
                        logdet_enclosure_gap: None,
                        consecutive_restored_incumbents: None,
                    })
                }
            })
        },
    );

    // The run may still fail — a rho whose cost is NaN at every later
    // iteration has nowhere to go — but HOW it fails is the contract. A
    // recoverable refusal must not be dressed as a fatal evaluation, because
    // `is_fatal_outer_evaluation()` is a hard `return Err(e)` in both the seed
    // loop and the strategy ladder, so the fatal label is what stops the
    // remaining seeds and the `disable_fixed_point` fallback plan from ever
    // being tried.
    let outcome = problem.run(&mut obj, "post-seed recoverable non-finite cost");

    // The NaN branch must actually have been reached, or the fixture proves
    // nothing: `efs_calls == 1` would mean only the descending seed ran.
    let calls = efs_calls.load(Ordering::Relaxed);
    assert!(
        calls >= 2,
        "fixture never reached the non-finite iteration: efs_calls={calls}"
    );

    match outcome {
        Err(error) => assert!(
            !error.is_fatal_outer_evaluation(),
            "a non-finite cost is the producer's own `Recoverable` verdict — it must not \
             short-circuit the seed loop and the fallback ladder as a fatal evaluation. \
             got: {error}"
        ),
        // A PASS HERE WOULD BE VACUOUS, so it is a failure. The contract under
        // test is how a recoverable refusal is CLASSIFIED, which only exists on
        // the error path; if the run returns Ok (e.g. `MaxIterationsReached`
        // hands back the last solution) the classification was never exercised
        // and this test has asserted nothing. Rebuild the fixture so the
        // refusal reaches `run_fixed_point_outer_solver`'s error arm rather
        // than relaxing this.
        Ok(result) => panic!(
            "the run SUCCEEDED, so the recoverable-vs-fatal classification was never \
             reached and this gate is vacuous (efs_calls={calls}, termination={:?})",
            result.termination
        ),
    }
}
