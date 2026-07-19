//! #2370 layer-2 (fail-loudly), the cases not covered by
//! [`super::super::run_plan::run_plan_tests::inverted_rho_box_is_a_typed_error_not_a_clamp_panic_2370`].
//!
//! That test already pins that an inverted box `[-10.0, -11.855…]` reaching
//! the runner is a typed `EstimationError::InvalidInput` rather than a
//! `f64::clamp` `min > max` panic escaping as an opaque `GamError`. The TYPE of
//! the refusal is not re-asserted here.
//!
//! What is left uncovered, and is what these four add:
//!
//! * That refusal must also stay as informative as the panic it replaced. The
//!   panic printed both bound values; a message naming only the coordinate is
//!   a weaker diagnostic, so the values are pinned explicitly.
//! * `f64::clamp` panics on a NaN bound just as it does on an inverted one, so
//!   non-finite walls need the same refusal.
//! * The guard reads the *effective* bounds template. A guard that wrongly
//!   rejected valid boxes would be a far worse regression than the panic it
//!   replaced, and nothing else asserts that ordinary problems still solve —
//!   so both an explicit ordered box and the implicit `±rho_bound` fallback
//!   need positive controls.

use super::*;
use ndarray::array;

fn quadratic_cost(_: &mut (), rho: &Array1<f64>) -> Result<f64, EstimationError> {
    Ok(0.5 * rho.dot(rho))
}

fn quadratic_eval(_: &mut (), rho: &Array1<f64>) -> Result<OuterEval, EstimationError> {
    Ok(OuterEval {
        cost: 0.5 * rho.dot(rho),
        gradient: rho.clone(),
        hessian: HessianValue::Dense(array![[1.0]]),
        inner_beta_hint: None,
    })
}

/// A strictly-convex outer objective minimized at ρ = 0.
fn quadratic_problem(bounds: Option<(Array1<f64>, Array1<f64>)>) -> OuterProblem {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_initial_rho(array![0.0])
        .with_problem_size(8, 3);
    match bounds {
        Some((lower, upper)) => problem.with_bounds(lower, upper),
        None => problem,
    }
}

macro_rules! quadratic_objective {
    ($problem:expr) => {
        $problem.build_objective(
            (),
            quadratic_cost,
            quadratic_eval,
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
    };
}

#[test]
fn the_inverted_box_refusal_carries_both_bound_values_2370() {
    // The reported geometry: the custom-family floor -10.0 against an
    // effective-df ceiling that drifted below it to -11.855…, i.e.
    // `f64::clamp(min = -10.0, max = -11.855421656441532)`.
    //
    // That the refusal is TYPED is pinned by
    // `run_plan_tests::inverted_rho_box_is_a_typed_error_not_a_clamp_panic_2370`.
    // What this pins is that it stays as DIAGNOSTIC as the panic it replaced:
    // the two bound values are what identify which pair of independently-owned
    // constants drifted, and they are what made #2370 localizable from the bug
    // report text alone. A coordinate index alone is a weaker diagnostic than
    // the panic, so asserting only on the index would quietly permit that
    // regression to return.
    let problem = quadratic_problem(Some((array![-10.0], array![-11.855_421_656_441_532])));
    let config = problem.config();
    let mut objective = quadratic_objective!(problem);

    let error = run_outer(&mut objective, &config, "inverted-box-2370")
        .expect_err("an inverted rho box must be refused, not clamped");
    let message = error.to_string();
    assert!(
        message.contains("-10") && message.contains("-11.855"),
        "the refusal must carry BOTH offending bound values, got: {message}"
    );
}

#[test]
fn a_non_finite_bound_is_a_typed_error_2370() {
    // `f64::clamp` panics when either bound is NaN, on the same
    // `min > max, or either was NaN` assertion that produced #2370. A
    // non-finite wall must therefore be refused on the same footing as an
    // inverted one.
    let problem = quadratic_problem(Some((array![-10.0], array![f64::NAN])));
    let config = problem.config();
    let mut objective = quadratic_objective!(problem);

    let error = run_outer(&mut objective, &config, "nonfinite-box-2370")
        .expect_err("a non-finite rho bound must be refused");
    assert!(
        matches!(error, EstimationError::InvalidInput(_)),
        "a non-finite rho bound must be EstimationError::InvalidInput, got: {error:?}"
    );
}

#[test]
fn an_ordered_box_still_solves_2370() {
    // Positive control: the optimum ρ = 0 is interior to the production box
    // [-10, 12], so the guard must pass it through and the solve must land.
    let problem = quadratic_problem(Some((array![-10.0], array![12.0])));
    let config = problem.config();
    let mut objective = quadratic_objective!(problem);

    let result = run_outer(&mut objective, &config, "ordered-box-2370")
        .expect("an ordered rho box must solve normally");
    assert!(
        result.rho[0].abs() < 1e-3,
        "the ordered-box solve must reach the interior optimum at rho=0, got {}",
        result.rho[0],
    );
}

#[test]
fn a_problem_with_no_explicit_box_still_solves_2370() {
    // The guard validates the EFFECTIVE template, which falls back to
    // `±config.rho_bound` when no explicit box is configured. That fallback is
    // ordered by construction, but it is a different code path from an
    // explicitly configured box and nothing else exercises it against the
    // guard.
    let problem = quadratic_problem(None);
    let config = problem.config();
    let mut objective = quadratic_objective!(problem);

    run_outer(&mut objective, &config, "default-box-2370")
        .expect("the default rho box must solve normally");
}
