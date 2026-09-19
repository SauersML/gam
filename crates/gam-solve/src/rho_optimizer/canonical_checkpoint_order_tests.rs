//! #2902: a refusal raised inside a permuted outer search publishes its
//! checkpoint in the caller's coordinate order.
//!
//! When structural keys put the ρ coordinates out of canonical order,
//! [`run_outer`] runs the whole search in canonical order and maps the result
//! back. The refusal took the other exit, through `?`, and so carried its
//! checkpoint in canonical order. The refusal names that vector as the point to
//! resume from, so the resumed search started at the permuted point, at a
//! criterion value the refusal never reported.
//!
//! The pin is the round trip a caller makes. A permuted search refuses. Its
//! printed checkpoint seeds the same search again. The first evaluation of the
//! resumed search must be at the checkpoint, and must score the refusal's own
//! `final_value`. The quartic's two coordinates carry different offsets and
//! weights, so swapping them moves the criterion (asserted below), and a
//! canonical-order checkpoint cannot pass.

use super::*;
use ndarray::array;

/// Off-lattice optimum, so no generated seed certifies at iteration 0 (the
/// same reasoning as `run_nonconverged_arc_returns_typed_checkpoint_without_a_budget_retry`).
const OFFSET: [f64; 2] = [0.5, -0.25];
/// Unequal weights keep the residual gradient far above the stationarity band
/// and make the criterion asymmetric under a coordinate swap.
const WEIGHT: [f64; 2] = [1.0e6, 3.0e6];

type Trail = Vec<(Array1<f64>, f64)>;

fn quartic(theta: &Array1<f64>) -> f64 {
    WEIGHT[0] * (theta[0] - OFFSET[0]).powi(4) + WEIGHT[1] * (theta[1] - OFFSET[1]).powi(4)
}

fn quartic_eval(theta: &Array1<f64>) -> OuterEval {
    let d = [theta[0] - OFFSET[0], theta[1] - OFFSET[1]];
    OuterEval {
        cost: quartic(theta),
        gradient: array![
            4.0 * WEIGHT[0] * d[0].powi(3),
            4.0 * WEIGHT[1] * d[1].powi(3)
        ],
        hessian: HessianValue::Dense(array![
            [12.0 * WEIGHT[0] * d[0].powi(2), 0.0],
            [0.0, 12.0 * WEIGHT[1] * d[1].powi(2)]
        ]),
        inner_beta_hint: None,
    }
}

fn permuted_search(initial_rho: Array1<f64>) -> OuterProblem {
    OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_initial_rho(initial_rho)
        .with_max_iter(1)
        // Keys out of order: the search runs with the two coordinates swapped.
        .with_rho_canonical_keys(Some(vec![20, 10]))
}

/// Run `problem` and return its outcome with every (θ, cost) the objective
/// was asked for, in call order and in the objective's own coordinates.
fn run_recording(problem: &OuterProblem) -> (Result<OuterResult, EstimationError>, Trail) {
    let mut obj = problem.build_objective(
        Trail::new(),
        |trail: &mut Trail, theta: &Array1<f64>| {
            let cost = quartic(theta);
            trail.push((theta.clone(), cost));
            Ok(cost)
        },
        |trail: &mut Trail, theta: &Array1<f64>| {
            let eval = quartic_eval(theta);
            trail.push((theta.clone(), eval.cost));
            Ok(eval)
        },
        None::<fn(&mut Trail)>,
        None::<fn(&mut Trail, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let outcome = problem.run(&mut obj, "permuted checkpoint order 2902");
    (outcome, obj.state)
}

#[test]
fn a_permuted_search_refusal_resumes_at_its_own_criterion_value_2902() {
    assert_eq!(
        canonical_permutation(&[20, 10]),
        Some(vec![1, 0]),
        "the keys must induce a non-identity search order, or nothing is permuted"
    );
    let refusal = run_recording(&permuted_search(array![5.0, -3.0]))
        .0
        .expect_err("a one-iteration budget on an off-lattice quartic must refuse");
    let EstimationError::RemlDidNotConverge {
        rho_checkpoint,
        final_value,
        reason,
        ..
    } = refusal
    else {
        panic!("expected typed REML non-convergence, got {refusal}");
    };
    // #2817: a coordinate the reason names is rendered in native order when the
    // text is written, so the refusal carries no canonical slot map for the caller
    // to decode (the native naming itself is pinned by
    // `native_coordinate_order_tests`).
    assert!(
        !reason.contains("canonical slot"),
        "the refusal must name coordinates natively, not append a canonical slot map; got: \
         {reason}"
    );
    let checkpoint = Array1::from_vec(rho_checkpoint);
    assert_eq!(checkpoint.len(), 2);
    let swapped = array![checkpoint[1], checkpoint[0]];
    assert_ne!(
        quartic(&swapped).to_bits(),
        quartic(&checkpoint).to_bits(),
        "the swap must move the criterion at checkpoint {checkpoint:?}, or this test cannot \
         tell the two orders apart"
    );

    let trail = run_recording(&permuted_search(checkpoint.clone())).1;
    let (first_theta, first_cost) = trail
        .first()
        .expect("the resumed search must evaluate the objective");
    assert_eq!(
        first_theta.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        checkpoint.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        "the resumed search must start at the printed checkpoint: first θ {first_theta:?}, \
         checkpoint {checkpoint:?}"
    );
    assert_eq!(
        first_cost.to_bits(),
        final_value.to_bits(),
        "the first evaluation at the printed checkpoint must score the refusal's own \
         final_value: got {first_cost:e}, refusal reported {final_value:e} (the swapped \
         point scores {:e})",
        quartic(&swapped)
    );
}
