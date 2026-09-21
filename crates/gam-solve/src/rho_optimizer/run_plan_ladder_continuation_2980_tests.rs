//! #2980: a plan change continues the search from the lowest state the attempt before it ended
//! on, not from the derived start.
//!
//! The fixture is a Gaussian well with its analytic curvature declared and a gradient-only
//! preference, so the ladder is BFGS followed by ARC on the declared curvature (#2898). Under a
//! budget one iteration short of what BFGS needs, the BFGS attempt stops inside the well without
//! converging and the ARC attempt follows it. Before #2980 the ARC attempt re-entered at the
//! derived start and re-descended the whole well: on a real fit that restart was over 60% of the
//! wall time (a survival marginal-slope fit, BFGS stationary at cost 3.650901e2, ARC back at
//! 3.849224e2 at its first exact-Hessian evaluation).

use super::*;
use ndarray::array;
use std::sync::{Arc, Mutex};

const CENTER: f64 = -3.5;
const WIDTH: f64 = 0.5;
const DEPTH: f64 = 10.0;
/// Inside the well's convex core and off its centre.
const START: f64 = -3.9;
/// `τ_stat = 1/(2n) = 5e-4`.
const N_OBS: usize = 1_000;

fn well_value(x: f64) -> f64 {
    -DEPTH * (-(x - CENTER).powi(2) / (2.0 * WIDTH * WIDTH)).exp()
}

fn well_derivative(x: f64) -> f64 {
    -well_value(x) * (x - CENTER) / (WIDTH * WIDTH)
}

fn well_curvature(x: f64) -> f64 {
    let u = x - CENTER;
    -well_value(x) * (1.0 / (WIDTH * WIDTH) - u * u / WIDTH.powi(4))
}

/// Every evaluation the objective served, in order: the order it was asked for and where.
type EvaluationLog = Arc<Mutex<Vec<(OuterEvalOrder, f64)>>>;

const POISONED: &str = "the evaluation log is only locked to push or copy, which cannot panic";

fn record(log: &EvaluationLog, order: OuterEvalOrder, rho: f64) {
    log.lock().expect(POISONED).push((order, rho));
}

fn run_well(
    max_iter: usize,
    prefer_gradient_only: bool,
    fallback: FallbackPolicy,
    label: &str,
) -> (Result<OuterResult, EstimationError>, Vec<(OuterEvalOrder, f64)>) {
    let (_cache_dir, session) = tmp_cache_session(label);
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_prefer_gradient_only(prefer_gradient_only)
        .with_fallback_policy(fallback)
        .with_bounds(array![-6.0], array![6.0])
        .with_initial_rho(array![START])
        .with_problem_size(N_OBS, 1)
        .with_max_iter(max_iter)
        .with_cache_session(session);
    let log: EvaluationLog = Arc::new(Mutex::new(Vec::new()));
    let eval = |rho: &Array1<f64>, order: OuterEvalOrder| OuterEval {
        cost: well_value(rho[0]),
        gradient: array![well_derivative(rho[0])],
        hessian: match order {
            OuterEvalOrder::ValueGradientHessian => {
                HessianValue::Dense(array![[well_curvature(rho[0])]])
            }
            OuterEvalOrder::Value | OuterEvalOrder::ValueAndGradient => HessianValue::Unavailable,
        },
        inner_beta_hint: None,
    };
    let mut objective = problem.build_objective_with_eval_order(
        Arc::clone(&log),
        |log: &mut EvaluationLog, rho: &Array1<f64>| {
            record(log, OuterEvalOrder::Value, rho[0]);
            Ok(well_value(rho[0]))
        },
        move |log: &mut EvaluationLog, rho: &Array1<f64>| {
            record(log, OuterEvalOrder::ValueGradientHessian, rho[0]);
            Ok(eval(rho, OuterEvalOrder::ValueGradientHessian))
        },
        move |log: &mut EvaluationLog, rho: &Array1<f64>, order: OuterEvalOrder| {
            record(log, order, rho[0]);
            Ok(eval(rho, order))
        },
        None::<fn(&mut EvaluationLog)>,
        None::<fn(&mut EvaluationLog, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let outcome = problem.run(&mut objective, label);
    let evaluations = log.lock().expect(POISONED).clone();
    (outcome, evaluations)
}

#[test]
fn the_declared_curvature_attempt_continues_from_the_gradient_only_stop_2980() {
    let (calibration, _) = run_well(200, true, FallbackPolicy::Disabled, "calibration #2980");
    let bfgs_iterations = calibration
        .expect("an unbounded BFGS search of the well from inside its core certifies its centre")
        .iterations;
    assert!(
        bfgs_iterations >= 2,
        "the BFGS search must take at least two iterations for a budget to split it; took \
         {bfgs_iterations}"
    );
    let (outcome, evaluations) = run_well(
        bfgs_iterations - 1,
        true,
        FallbackPolicy::Automatic,
        "ladder continuation #2980",
    );
    let first_second_order = evaluations
        .iter()
        .position(|(order, _)| *order == OuterEvalOrder::ValueGradientHessian)
        .expect("the ladder must reach the declared-curvature attempt");
    let (_, last_gradient_only_rho) = evaluations[first_second_order - 1];
    assert_eq!(
        evaluations[first_second_order].1.to_bits(),
        last_gradient_only_rho.to_bits(),
        "the ARC attempt must open at the point the BFGS attempt stopped on, not at the derived \
         start; evaluations={evaluations:?}"
    );
    assert_eq!(
        evaluations.iter().filter(|(_, rho)| rho.to_bits() == START.to_bits()).count(),
        1,
        "only the BFGS attempt's opening evaluation may sit at the derived start; \
         evaluations={evaluations:?}"
    );
    let published = outcome.unwrap_or_else(|error| {
        panic!("the ARC attempt certifies the well's centre and must publish: {error}")
    });
    assert!(
        (published.rho[0] - CENTER).abs() < 1.0e-3,
        "the published optimum must be the well's centre; rho={:?}",
        published.rho
    );
}
