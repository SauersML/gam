//! #3306 — a degraded plan resumes the lowest state an earlier attempt of the
//! same ladder ended at.
//!
//! The gradient-only BFGS search of the #2898 lifecycle stops under its
//! iteration budget without a claim. The ladder then retries with exact
//! curvature. That retry used to start again at the configured seed and
//! receive BFGS's incumbent only as a comparator, so every accepted BFGS step
//! was discarded: on a binary Bernoulli marginal-slope fit BFGS had reached the
//! certified value to seven digits in ~380 s, and ARC re-searched from the seed.
//!
//! The pin is a curved valley `u = ρ₀ − ρ₁²`,
//! `f = ½·wall·u² + ½·(ρ₁ − 1)²`, whose BFGS walk does not finish in two
//! iterations from `(3, −1)`.

use super::*;
use ndarray::array;

const WALL: f64 = 1.0e2;
const TARGET: f64 = 1.0;

fn valley_eval(rho: &Array1<f64>) -> OuterEval {
    let r1 = rho[1];
    let u = rho[0] - r1 * r1;
    let cross = -2.0 * WALL * r1;
    OuterEval {
        cost: 0.5 * WALL * u * u + 0.5 * (r1 - TARGET) * (r1 - TARGET),
        gradient: array![WALL * u, cross * u + (r1 - TARGET)],
        hessian: HessianValue::Dense(array![
            [WALL, cross],
            [cross, WALL * (4.0 * r1 * r1 - 2.0 * u) + 1.0],
        ]),
        inner_beta_hint: None,
    }
}

/// Every evaluation the search requested, tagged with the plan attempt that
/// requested it. `reset` runs once at the start of each attempt.
#[derive(Default)]
struct Trace {
    attempt: usize,
    evaluations: Vec<(usize, Array1<f64>, f64)>,
}

impl Trace {
    fn record(&mut self, rho: &Array1<f64>) -> OuterEval {
        let eval = valley_eval(rho);
        self.evaluations.push((self.attempt, rho.clone(), eval.cost));
        eval
    }
}

#[test]
fn the_exact_curvature_retry_resumes_the_gradient_only_incumbent_3306() {
    let seed = array![3.0, -1.0];
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_prefer_gradient_only(true)
        .with_bounds(Array1::from_elem(2, -20.0), Array1::from_elem(2, 20.0))
        .with_initial_rho(seed.clone())
        .with_max_iter(2);
    let mut obj = problem.build_objective(
        Trace::default(),
        |trace: &mut Trace, rho: &Array1<f64>| Ok(trace.record(rho).cost),
        |trace: &mut Trace, rho: &Array1<f64>| Ok(trace.record(rho)),
        Some(|trace: &mut Trace| trace.attempt += 1),
        None::<fn(&mut Trace, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    // The run's own result is not what this pins: under a two-iteration budget the
    // retry may still refuse. What it must not do is start over at the seed.
    let outcome = problem.run(&mut obj, "gradient-only incumbent resume #3306");
    let trace = &obj.state;
    let attempts: Vec<usize> = {
        let mut seen: Vec<usize> = trace.evaluations.iter().map(|(a, _, _)| *a).collect();
        seen.dedup();
        seen
    };
    assert!(
        attempts.len() >= 2,
        "the gradient-only attempt must stop unclaimed under a two-iteration budget and hand \
         over to the exact-curvature retry; attempts seen: {attempts:?}, outcome: {:?}",
        outcome.as_ref().map(|result| result.plan_used.solver)
    );
    let (first_attempt, retry_attempt) = (attempts[0], attempts[1]);
    let seed_cost = valley_eval(&seed).cost;
    let first_attempt_lowest = trace
        .evaluations
        .iter()
        .filter(|(a, _, _)| *a == first_attempt)
        .map(|(_, _, cost)| *cost)
        .fold(f64::INFINITY, f64::min);
    assert!(
        first_attempt_lowest < seed_cost,
        "the gradient-only attempt must descend from the seed ({seed_cost:e}) before it stops; \
         lowest {first_attempt_lowest:e}"
    );
    let (_, retry_start, retry_start_cost) = trace
        .evaluations
        .iter()
        .find(|(a, _, _)| *a == retry_attempt)
        .expect("the retry attempt evaluated");
    assert_ne!(
        retry_start, &seed,
        "the exact-curvature retry restarted at the configured seed and discarded the \
         gradient-only attempt's incumbent"
    );
    assert!(
        trace
            .evaluations
            .iter()
            .any(|(a, rho, _)| *a == first_attempt && rho == retry_start),
        "the retry must start at a state the gradient-only attempt evaluated; started at \
         {retry_start:?}"
    );
    assert!(
        *retry_start_cost < seed_cost,
        "the retry must start at the gradient-only attempt's descended incumbent \
         (cost {retry_start_cost:e} against seed cost {seed_cost:e})"
    );
}
