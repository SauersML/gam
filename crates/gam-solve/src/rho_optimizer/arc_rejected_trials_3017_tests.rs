// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): the cost-stall guard on the trust-region routes counts accepted
// iterates, not evaluated trials (#3017). Scope comes from the parent via
// `use super::*`.
//
// The defect these pin. `opt::Arc` evaluates the objective at every trial
// `x_k + s` and only then runs the ratio test. Both trust-region bridges folded
// that evaluation into the cost-stall guard as though it were the next
// iterate, so a run of rejections — ARC raising σ toward the Hessian's
// Lipschitz scale — read as a window of accepted steps that bought nothing,
// and the guard stopped the run with `OUTER_ARC_UNPROGRESSING_STALL` at the
// seed.

use super::*;
use ndarray::array;

/// The criterion of #3017's fixture: `V(ρ) = c·e^ρ − (a − 1)·ρ`, the shape an
/// informative gamma hyperprior on a precision gives the log-precision. Its
/// curvature `c·e^ρ` grows exponentially, so a Newton step from below the
/// optimum overshoots and ARC must shrink it through several rejections.
const C_3017: f64 = 100.0;
const A_MINUS_ONE_3017: f64 = 100_000.0;

/// Where the issue's fit started: the Newton step from here, `3.344`, lands at
/// `ρ = 8.78`, where the criterion is `−2.26e5` against the seed's `−5.208e5`.
const SEED_3017: f64 = 5.4388;

fn value_3017(rho: f64) -> f64 {
    C_3017 * rho.exp() - A_MINUS_ONE_3017 * rho
}

fn gradient_3017(rho: f64) -> f64 {
    C_3017 * rho.exp() - A_MINUS_ONE_3017
}

fn hessian_3017(rho: f64) -> f64 {
    C_3017 * rho.exp()
}

/// The optimum: `ρ* = ln((a − 1)/c) = 6.9078`.
fn optimum_3017() -> f64 {
    (A_MINUS_ONE_3017 / C_3017).ln()
}

/// The whole outer run on the #3017 criterion reaches its optimum, and ARC
/// from the seed reaches it itself. Before the repair the guard folded each of
/// ARC's rejected trials as a non-improving step, filled two windows at the
/// seed after six trials and stopped the seed's run there with
/// `OUTER_ARC_UNPROGRESSING_STALL`, `1.47` short of the optimum.
#[test]
fn arc_reaches_the_optimum_through_a_run_of_rejected_trials_3017() {
    #[derive(Default)]
    struct State {
        evaluated: Vec<f64>,
    }

    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![SEED_3017])
        .with_bounds(array![-30.0], array![30.0]);
    let mut obj = problem.build_objective(
        State::default(),
        |_: &mut State, theta: &Array1<f64>| Ok(value_3017(theta[0])),
        |state: &mut State, theta: &Array1<f64>| {
            state.evaluated.push(theta[0]);
            Ok(OuterEval {
                cost: value_3017(theta[0]),
                gradient: array![gradient_3017(theta[0])],
                hessian: HessianValue::Dense(array![[hessian_3017(theta[0])]]),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut State)>,
        None::<fn(&mut State, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let config = problem.config();
    let cap = obj.capability();
    let the_plan = plan(&cap);
    assert_eq!(the_plan.solver, Solver::Arc, "the fixture must run the dense ARC route");
    let outcome = run_outer_with_plan(&mut obj, &config, "#3017 rejected trials", &cap, &the_plan, true)
        .expect("the #3017 criterion has an interior optimum the run must reach");
    let result = match outcome {
        PlanRunOutcome::Converged(result) => result,
        other => panic!(
            "the run must converge, not stop short: {:?}; evaluated {:?}",
            std::mem::discriminant(&other),
            obj.state.evaluated
        ),
    };
    let rho = result.rho[0];
    let evaluated = &obj.state.evaluated;
    assert!(
        (rho - optimum_3017()).abs() < 1.0e-3,
        "the run must end at ρ* = {:.6}, not at {rho:.6}; evaluated {evaluated:?}",
        optimum_3017(),
    );
    // The seed's own ARC trajectory: every trial from the seed toward its
    // overshooting first trial, before the seed loop leaves for another start.
    // The defect stopped this trajectory at the seed, six trials in, and the
    // run reached ρ* only because a later lattice start happened to.
    let first_trial = evaluated[1];
    let seed_trajectory: Vec<f64> = evaluated
        .iter()
        .copied()
        .take_while(|&trial| (SEED_3017..=first_trial).contains(&trial))
        .collect();
    assert!(
        seed_trajectory
            .iter()
            .any(|&trial| (trial - optimum_3017()).abs() < 1.0e-3),
        "ARC from the seed must itself reach ρ* = {:.6} through its rejected trials; \
         its trajectory was {seed_trajectory:?}",
        optimum_3017(),
    );
    let rejected = seed_trajectory
        .iter()
        .filter(|&&trial| value_3017(trial) > value_3017(SEED_3017))
        .count();
    assert!(
        rejected > ARC_COST_STALL_WINDOW,
        "the fixture must spend more trials above the seed than one stall window, or \
         it does not exercise the defect: {rejected} over {seed_trajectory:?}",
    );
}

/// Drive the dense ARC bridge from the #3017 seed through `trials`, each one
/// above the seed's criterion. `report_accepted` says whether `opt` accepted
/// them. Returns the outcome of every evaluation up to the first stop.
fn drive_trials_3017(trials: &[f64], report_accepted: bool) -> Vec<Result<f64, String>> {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let config = problem.config();
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(value_3017(theta[0])),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        |_: &mut (), theta: &Array1<f64>, order: OuterEvalOrder| {
            Ok(OuterEval {
                cost: value_3017(theta[0]),
                gradient: array![gradient_3017(theta[0])],
                hessian: match order {
                    OuterEvalOrder::ValueGradientHessian => {
                        HessianValue::Dense(array![[hessian_3017(theta[0])]])
                    }
                    _ => HessianValue::Unavailable,
                },
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    // The guard exactly as the dense ARC route builds and seeds it.
    let rel_tol = config
        .rel_cost_tolerance
        .unwrap_or(config.tolerance * 1.0e-2)
        .max(COST_STALL_REL_TOL_FLOOR);
    let mut guard = CostStallGuard::new(rel_tol, ARC_COST_STALL_WINDOW, &config, exit);
    guard.observe_second_order_seed(
        &array![SEED_3017],
        value_3017(SEED_3017),
        gradient_3017(SEED_3017).abs(),
        Some(true),
    );
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        hessian_source: HessianSource::Analytic,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some((array![-30.0], array![30.0])),
        curvature_stationary_floor: Some(outer_rel_cost_floor(&config)),
        accepted_trials: AcceptedTrialGate::new(Arc::clone(&ledger)),
        decrement_verdict_config: Some(&config),
    };
    let mut outcomes = Vec::new();
    for (iter, &trial) in trials.iter().enumerate() {
        let outcome = SecondOrderObjective::eval_hessian(&mut bridge, &array![trial])
            .and_then(|sample| {
                if report_accepted {
                    report_accepted_trial_3017(&ledger, iter);
                }
                bridge.settle_pending_trial().map_or(Ok(sample.value), Err)
            });
        let stopped = outcome.is_err();
        outcomes.push(outcome.map_err(|err| err.into_message()));
        if stopped {
            break;
        }
    }
    outcomes
}

/// The guard counts ARC's iterates, not its trials. Two stall windows of
/// rejected trials at one iterate are σ growing toward the curvature's scale,
/// not steps that bought nothing, so they never stop the run. The control arm
/// feeds the same trials as accepted steps, which is what the bridge did with
/// every trial before the repair, and the guard does stop there.
#[test]
fn arc_bridge_rejected_trials_never_fill_the_stall_window_3017() {
    // The issue's rejected trials, shrinking from the Newton overshoot at 8.78;
    // each stays above the seed's criterion.
    let trials: Vec<f64> = (0..2 * ARC_COST_STALL_WINDOW + 2)
        .map(|k| SEED_3017 + 3.3443 - 0.05 * k as f64)
        .collect();
    assert!(
        trials
            .iter()
            .all(|&trial| value_3017(trial) > value_3017(SEED_3017)),
        "every scripted trial must be one ARC's ratio test rejects: {trials:?}"
    );

    let rejected = drive_trials_3017(&trials, false);
    assert_eq!(rejected.len(), trials.len(), "no rejected trial may stop the run");
    assert!(
        rejected.iter().all(Result::is_ok),
        "rejected trials must never stop the run at the seed: {rejected:?}"
    );

    let accepted = drive_trials_3017(&trials, true);
    assert_eq!(
        accepted.last(),
        Some(&Err(ARC_UNPROGRESSING_STALL_SENTINEL.to_string())),
        "the same trials as accepted non-improving iterates fill two windows and \
         stop the run: {accepted:?}"
    );
}
