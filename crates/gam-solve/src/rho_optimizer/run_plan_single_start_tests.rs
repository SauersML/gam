// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): the outer search's one derived start. Every run enters from exactly
// one point — the caller's `initial_rho`, else the problem's data-derived
// heuristic, else ρ=0 — projected into the box, and follows one trajectory
// from it. Child modules see the parent's entire scope via `use super::*`.

use super::*;
use ndarray::array;

fn quadratic_bowl_objective(
    problem: &OuterProblem,
    center: Array1<f64>,
    seen: Arc<Mutex<Vec<Array1<f64>>>>,
) -> impl OuterObjective {
    let cost_center = center.clone();
    let cost_seen = Arc::clone(&seen);
    problem.build_objective(
        (),
        move |_: &mut (), theta: &Array1<f64>| {
            cost_seen.lock().expect("cost log lock").push(theta.clone());
            let d = theta - &cost_center;
            Ok(0.5 * d.dot(&d))
        },
        move |_: &mut (), theta: &Array1<f64>| {
            seen.lock().expect("evaluation log lock").push(theta.clone());
            let d = theta - &center;
            Ok(OuterEval {
                cost: 0.5 * d.dot(&d),
                gradient: d,
                hessian: HessianValue::Dense(Array2::eye(theta.len())),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    )
}

/// The outer search runs ONE trajectory from ONE start. Before the single-start
/// refactor the run generated a seed lattice around the start and, on a
/// Gaussian-profile problem, drove every certified seed to the same optimum
/// (speed.md F3: three ARC seeds converging to one ρ̂ cost ~60% of the outer
/// time). Here the first evaluation is the declared start, and once the
/// trajectory reaches the optimum no later evaluation jumps back out to a
/// distant restart point.
#[test]
fn outer_search_follows_one_trajectory_from_one_start() {
    let center = array![1.0, -1.0, 0.5];
    let start = array![4.0, 4.0, -4.0];
    let seen = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(3)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_initial_rho(start.clone())
        .with_max_iter(64);
    let mut obj = quadratic_bowl_objective(&problem, center.clone(), Arc::clone(&seen));
    let result = problem
        .run(&mut obj, "single outer start")
        .expect("a convex quadratic certifies from its one start");
    let dist = |p: &Array1<f64>| (p - &center).mapv(|v| v * v).sum().sqrt();
    assert!(dist(&result.rho) < 1e-4, "converged rho={:?}", result.rho);
    let seen = seen.lock().expect("evaluation log lock");
    assert_eq!(seen.first(), Some(&start), "the first evaluation is the declared start");
    let reached = seen
        .iter()
        .position(|p| dist(p) < 1e-3)
        .expect("the trajectory reaches the optimum");
    let start_dist = dist(&start);
    let farthest_after = seen[reached..].iter().map(dist).fold(0.0_f64, f64::max);
    assert!(
        farthest_after < 0.5 * start_dist,
        "after reaching the optimum the run evaluated a point {farthest_after:.3} away — a \
         restart, not one trajectory (start distance {start_dist:.3}); trace={seen:?}",
    );
}

/// With no caller start, the run enters from the problem's data-derived
/// heuristic start, projected into the box, and nowhere else first.
#[test]
fn outer_search_enters_from_the_projected_heuristic_start() {
    let center = array![0.5, 0.5];
    let seen = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_bounds(array![-2.0, -2.0], array![2.0, 2.0])
        .with_heuristic_log_lambdas(vec![1.5, 9.0])
        .with_max_iter(64);
    let mut obj = quadratic_bowl_objective(&problem, center, Arc::clone(&seen));
    problem
        .run(&mut obj, "heuristic outer start")
        .expect("a convex quadratic certifies from its one start");
    assert_eq!(
        seen.lock().expect("evaluation log lock").first().cloned(),
        Some(array![1.5, 2.0]),
        "the heuristic start is clamped into the box and evaluated first",
    );
}

#[test]
fn run_starts_solver_with_direct_startup_eval() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let calls = Arc::clone(&calls);
            move |_: &mut (), theta: &Array1<f64>| {
                calls.lock().expect("call log lock").push("cost");
                Ok(theta[0] * theta[0])
            }
        },
        {
            let calls = Arc::clone(&calls);
            move |_: &mut (), theta: &Array1<f64>| {
                calls.lock().expect("call log lock").push("eval");
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: HessianValue::Dense(array![[2.0]]),
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    // This test pins the STARTUP eval ORDER, not convergence. The single-iter
    // budget leaves a small residual gradient above the tight stationarity bound,
    // so the run may legitimately refuse to certify — that outcome is orthogonal
    // to what is asserted here. The `calls` trace records the startup sequence
    // whether or not the run mints, so the run's Result is deliberately ignored.
    drop(problem.run(&mut obj, "solver should start from a direct startup eval"));
    let calls = calls.lock().expect("call log lock");
    let first_eval_idx = calls
        .iter()
        .position(|call| *call == "eval")
        .expect("solver should eventually request a full eval");
    assert!(
        first_eval_idx == 0,
        "startup should not perform a separate cost-screening pass first: {calls:?}"
    );
}

#[test]
fn run_typed_efs_runtime_fallback_degrades_to_bfgs_immediately() {
    let efs_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(12)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(Array1::zeros(12))
        .with_max_iter(5);
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
            Some(move |_: &mut (), _: &Array1<f64>| {
                efs_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // The EFS bridge translates typed gradient unavailability into
                // a typed first-order fallback request; no message token
                // participates in routing.
                Err(EstimationError::GradientUnavailable {
                    context: "synthetic EFS runtime escape hatch",
                    mode: "efs runtime escape hatch",
                })
            })
        },
    );
    let result = problem
        .run(&mut obj, "EFS runtime fallback request")
        .expect("runtime EFS escape hatch should degrade to BFGS");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    assert_eq!(
        efs_calls.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "runtime fallback request should abort the EFS attempt immediately"
    );
}

#[test]
fn run_rejects_invalid_theta_layout() {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_psi_dim(2)
        .with_initial_rho(Array1::zeros(1))
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(0.0),
        |_: &mut (), _: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.0,
                gradient: Array1::zeros(1),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let err = problem
        .run(&mut obj, "test invalid layout")
        .expect_err("invalid theta layout should fail cleanly");
    assert!(
        err.to_string().contains("invalid outer theta layout"),
        "unexpected error: {err}"
    );
}

#[test]
fn run_arc_projects_seed_before_seed_validation_eval() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_bounds(array![0.0], array![1.0])
        .with_initial_rho(array![2.0])
        // The subject is WHICH point the first evaluation sees, and that is
        // settled before the optimizer takes any step — `seen.first()` proves
        // the ordering on its own. The iteration budget is incidental here, and
        // at `1` it was actively harmful: one ARC step from the projected seed
        // `1.0` toward the optimum `0.25` lands at `0.2504` with
        // `|Pg| = 8.541e-4` against a `6.325e-4` stationarity bound — 1.35×
        // short — so the run refused ("claimed_converged=false after 1 outer
        // iteration(s)") and the `expect` below fired on a convergence budget
        // that has nothing to do with seed projection. Give the quadratic room
        // to certify, so projection is the only thing that can fail here.
        .with_max_iter(16);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok((theta[0] - 0.25).powi(2)),
        {
            let seen = Arc::clone(&seen);
            move |_: &mut (), theta: &Array1<f64>| {
                seen.lock().expect("evaluation log lock").push(theta.clone());
                Ok(OuterEval {
                    cost: (theta[0] - 0.25).powi(2),
                    gradient: array![2.0 * (theta[0] - 0.25)],
                    hessian: HessianValue::Dense(array![[2.0]]),
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem
        .run(&mut obj, "arc seed projection")
        .expect("arc should evaluate the projected seed");
    assert_eq!(
        seen.lock().expect("evaluation log lock").first().cloned(),
        Some(array![1.0]),
        "Arc must project the seed before validating the initial sample",
    );
}

#[test]
fn run_bfgs_projects_seed_before_seed_validation_eval() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![0.0], array![1.0])
        .with_initial_rho(array![2.0])
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok((theta[0] - 0.25).powi(2)),
        {
            let seen = Arc::clone(&seen);
            move |_: &mut (), theta: &Array1<f64>| {
                seen.lock().expect("evaluation log lock").push(theta.clone());
                Ok(OuterEval {
                    cost: (theta[0] - 0.25).powi(2),
                    gradient: array![2.0 * (theta[0] - 0.25)],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    // This test pins seed PROJECTION (the initial ρ=[2.0] is clamped to the box
    // upper bound [1.0] before the first sample eval), not convergence. The
    // single-iter budget from the projected seed need not reach the [0.25]
    // optimum, so the run may legitimately refuse to certify — orthogonal to the
    // projection assertion. The `seen` trace records the first evaluated point
    // whether or not the run mints, so the run's Result is deliberately ignored.
    drop(problem.run(&mut obj, "bfgs seed projection"));
    assert_eq!(
        seen.lock().expect("evaluation log lock").first().cloned(),
        Some(array![1.0]),
        "BFGS must project the seed before validating the initial sample",
    );
}
