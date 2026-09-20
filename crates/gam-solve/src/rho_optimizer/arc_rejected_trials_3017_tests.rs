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

/// The observations the whole-run fixture declares, the scale of the
/// hyperprior's shape: `τ_stat = 1/(2n) = 5e-6`.
const N_OBS_3017: usize = 100_000;

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

/// The whole outer run on the #3017 criterion certifies its optimum, and ARC
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
        .with_bounds(array![-30.0], array![30.0])
        .with_problem_size(N_OBS_3017, 1);
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
    assert_eq!(plan(&cap).solver, Solver::Arc, "the fixture must run the dense ARC route");
    let result = run_outer(&mut obj, &config, "#3017 rejected trials")
        .expect("the #3017 criterion has an interior optimum the run must reach");
    assert!(
        result.converged(),
        "the run must certify its optimum, not stop short; evaluated {:?}",
        obj.state.evaluated
    );
    let rho = result.rho[0];
    let evaluated = &obj.state.evaluated;
    assert!(
        (rho - optimum_3017()).abs() < 1.0e-3,
        "the run must end at ρ* = {:.6}, not at {rho:.6}; evaluated {evaluated:?}",
        optimum_3017(),
    );
    // The seed's own ARC trajectory: every trial from the seed toward its
    // overshooting first trial, before any later evaluation of the run. The
    // defect stopped this trajectory at the seed, six trials in.
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
    // Two stalls at one incumbent that bought nothing stop a run (#3018), so the
    // defect needs two trials above the seed.
    assert!(
        rejected >= 2,
        "the fixture must spend at least two trials above the seed, or it does not \
         exercise the defect: {rejected} over {seed_trajectory:?}",
    );
}

/// The same criterion and seed with no declared problem size, so no criterion
/// resolution and a solver band of `1e-3` (#3286). ARC reaches `ρ*` to within
/// `3.9e-7`, where `|g| = 3.9e-2` and `H = 1e5` still leave `½g²/H = 7.6e-9` of
/// decrease, about 50× the criterion's rounding. Its Newton step there is
/// `3.9e-7`, but the point `fl(ρ + s)` it lands on is off the step by up to
/// `ulp(6.9)/2 = 4.4e-16`. The model gradient at the represented step is then
/// `H·r ≈ 4.4e-11`, while opt's termination test asked for `θ‖s‖² ≈ 1.5e-13`,
/// so every trial was refused before evaluation, σ climbed to its ceiling and
/// the run ended on `trust_region_reject_floor` at `|g| = 3.9e-2`.
#[test]
fn arc_takes_the_newton_step_whose_represented_point_rounds_3286() {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_tolerance(1.0e-3)
        .with_initial_rho(array![SEED_3017]);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(value_3017(theta[0])),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: value_3017(theta[0]),
                gradient: array![gradient_3017(theta[0])],
                hessian: HessianValue::Dense(array![[hessian_3017(theta[0])]]),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let cap = obj.capability();
    assert_eq!(plan(&cap).solver, Solver::Arc, "the fixture must run the dense ARC route");
    let result = problem
        .run(&mut obj, "#3286 represented step")
        .unwrap_or_else(|error| panic!("ARC must certify ρ* = {:.6}: {error}", optimum_3017()));
    assert!(result.converged(), "the run must certify its optimum");
    let gradient = gradient_3017(result.rho[0]).abs();
    assert!(
        gradient <= 1.0e-3,
        "the certified point must meet the solver band: |g| = {gradient:.3e} at ρ = {:.12}",
        result.rho[0]
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
    // The guard exactly as the dense ARC route builds and seeds it. The scripted
    // criterion publishes no evidence, so its values carry the criterion's
    // resolution (#3018).
    let resolution = outer_criterion_resolution(&config);
    let mut guard = CostStallGuard::new(resolution, &config, exit);
    guard.observe_second_order_seed(
        &array![SEED_3017],
        value_3017(SEED_3017),
        resolution,
        gradient_3017(SEED_3017).abs(),
        Some(true),
    );
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        hessian_source: HessianSource::Analytic,
        eval_count: 0,
        inner_progress: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some((array![-30.0], array![30.0])),
        curvature_stationary_resolution: Some(resolution),
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

/// The guard counts ARC's iterates, not its trials. Eight rejected trials at one
/// iterate are σ growing toward the curvature's scale, not steps that bought
/// nothing, so they never stop the run. The control arm
/// feeds the same trials as accepted steps, which is what the bridge did with
/// every trial before the repair, and the guard does stop there.
#[test]
fn arc_bridge_rejected_trials_never_fill_the_stall_window_3017() {
    // The issue's rejected trials, shrinking from the Newton overshoot at 8.78;
    // each stays above the seed's criterion.
    let trials: Vec<f64> = (0..8)
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
        "the same trials as accepted non-improving iterates stall twice and stop \
         the run: {accepted:?}"
    );
}

/// A 1×1 Hessian served only as a product, so the plan runs opt's matrix-free
/// trust region. Counts its products, which only that route forms.
struct ScalarHessianProduct3017 {
    curvature: f64,
    products: Arc<AtomicUsize>,
}

impl HessianOperator for ScalarHessianProduct3017 {
    fn dim(&self) -> usize {
        1
    }

    fn apply_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<(), ObjectiveEvalError> {
        self.products.fetch_add(1, Ordering::Relaxed);
        out[0] = self.curvature * v[0];
        Ok(())
    }
}

/// `V(ρ) = k·(β·ρ⁴/4 − ρ)`: flat at `ρ₀ = 0` (`H₀ = 0`), then steep, which is
/// what a trust region's quadratic model misjudges over the widest range of
/// radii. Minimized at `ρ* = β^(−1/3)`.
const K_TR_3017: f64 = 1_000.0;
const BETA_TR_3017: f64 = 1.0e12;

/// The matrix-free trust region evaluates `(f, g, Hv)` at every trial and
/// rejects the ones its ratio test fails, as ARC does, so the same stall
/// window cut it off (#3017).
///
/// With `H₀ = 0` the model is linear, so the Steihaug step runs to the
/// boundary: `s = r`, predicted decrease `k·r`, actual `k·(r − β·r⁴/4)`. The
/// ratio `1 − β·r³/4` stays below `η = 0.1` while `r > (3.6/β)^(1/3) = 1.53e-4`.
/// From opt's initial radius 1, quartered per rejection, the seven trials
/// `r = 4^(−j)`, `j = 0..=6`, are rejected: two full three-trial stall windows
/// and one more, before #3018 deleted the window; a rejected trial reaches no
/// verdict. `r = 6.1e-5` is then accepted
/// (ratio 0.943), and the run must go on to `ρ* = 1e-4`.
#[test]
fn rejected_trust_region_trials_do_not_fill_the_stall_window_3017() {
    let rho_star = BETA_TR_3017.powf(-1.0 / 3.0);
    let cost = |rho: &Array1<f64>| K_TR_3017 * (BETA_TR_3017 * rho[0].powi(4) / 4.0 - rho[0]);
    let products = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Operator {
            materialization: HessianMaterialization::Unavailable,
            estimated_materialization_cost: None,
        })
        .with_tolerance(1.0e-3)
        .with_initial_rho(array![0.0]);
    let counted = Arc::clone(&products);
    let mut obj = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(cost(rho)),
        move |_: &mut (), rho: &Array1<f64>| {
            Ok(OuterEval {
                cost: cost(rho),
                gradient: array![K_TR_3017 * (BETA_TR_3017 * rho[0].powi(3) - 1.0)],
                hessian: HessianValue::Operator(Arc::new(ScalarHessianProduct3017 {
                    curvature: K_TR_3017 * 3.0 * BETA_TR_3017 * rho[0].powi(2),
                    products: Arc::clone(&counted),
                })),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "tr-rejected-trials-3017")
        .unwrap_or_else(|error| panic!("the fit must reach ρ*={rho_star:.4e}: {error}"));
    assert!(
        products.load(Ordering::Relaxed) > 0,
        "an operator-only Hessian must run the matrix-free trust region"
    );
    let reached = result.rho[0];
    assert!(
        (reached / rho_star - 1.0).abs() < 1e-2,
        "the trust region must reach ρ*={rho_star:.4e}, not stop at ρ₀=0: ρ={reached:.4e}"
    );
}
