// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): #3279, an ARC claim on a trial above the iterate the run stood on.
//
// opt's ARC ends on a trial whose projected gradient clears its tolerance before
// the ratio test compares the trial with the current iterate. Where the
// criterion flattens at a height, the λ→∞ face of a REML criterion, such a trial
// is stationary and uphill at once. The fixture is a Gaussian well with a
// sigmoid step to a plateau above it: the search starts on the well's concave
// shoulder, where the descent direction runs up the step, and a small cubic
// regularisation sends its first trial onto the plateau at the upper bound. The
// run must not publish the plateau; the start beats it, so it is the attempt's
// checkpoint and the claim is declined as a dominated plateau.

use super::*;
use ndarray::array;

const WELL_CENTER: f64 = 1.0;
const STEP_CENTER: f64 = 6.0;
/// On the well's concave shoulder: the slope descends toward the step.
const SHOULDER_START: f64 = -0.5;
const UPPER: f64 = 60.0;

fn well_with_plateau(x: f64) -> (f64, f64, f64) {
    let u = x - WELL_CENTER;
    let well = (-0.5 * u * u).exp();
    let t = (x - STEP_CENTER).tanh();
    let sech_sq = 1.0 - t * t;
    (
        -well + 1.0 + t,
        u * well + sech_sq,
        (1.0 - u * u) * well - 2.0 * t * sech_sq,
    )
}

#[test]
fn an_arc_claim_above_the_iterate_it_left_is_declined_as_a_dominated_plateau_3279() {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_bounds(array![-5.0], array![UPPER])
        .with_initial_rho(array![SHOULDER_START]);
    let mut objective = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(well_with_plateau(rho[0]).0),
        |_: &mut (), rho: &Array1<f64>| {
            let (cost, gradient, hessian) = well_with_plateau(rho[0]);
            Ok(OuterEval {
                cost,
                gradient: array![gradient],
                hessian: HessianValue::Dense(array![[hessian]]),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut config = problem.config();
    // A cubic regularisation well below the shoulder's curvature: the model's
    // step along the negative curvature is |H|/σ, far past the upper bound.
    config.arc_initial_regularization = Some(1e-8);
    let label = "ARC uphill stationary trial #3279";
    let capability = primary_capability_for_config(objective.capability(), &config, label);
    let the_plan = plan(&capability);
    assert!(
        matches!(the_plan.solver, Solver::Arc),
        "fixture precondition: a dense analytic Hessian plans ARC, got {the_plan}"
    );
    let (start_value, start_gradient, start_curvature) = well_with_plateau(SHOULDER_START);
    assert!(
        start_gradient < 0.0 && start_curvature < 0.0,
        "fixture precondition: the start descends toward the step on negative curvature"
    );
    let (plateau_value, plateau_gradient, _) = well_with_plateau(UPPER);
    assert!(
        plateau_value > start_value + 1.0 && plateau_gradient.abs() < 1e-40,
        "fixture precondition: the upper bound is a stationary plateau above the start"
    );

    let outcome = run_outer_with_plan(&mut objective, &config, label, &capability, &the_plan, false)
        .expect("the attempt ends with an outcome");
    match outcome {
        PlanRunOutcome::DominatedPlateau(dominated) => {
            assert!(
                (dominated.plateau.final_value - plateau_value).abs() < 1e-9,
                "the declined claim is the plateau trial: {:.6e}",
                dominated.plateau.final_value
            );
            assert!(
                dominated.incumbent.final_value <= start_value,
                "the checkpoint is the lowest iterate the run accepted, at or below the start: \
                 {:.6e} vs {start_value:.6e}",
                dominated.incumbent.final_value
            );
            assert!(!dominated.incumbent.solver_claimed_convergence());
        }
        PlanRunOutcome::Converged(result) => {
            assert!(
                result.final_value <= start_value,
                "the attempt published rho={:?} at {:.6e}, above the start's {start_value:.6e}",
                result.rho.to_vec(),
                result.final_value
            );
        }
        PlanRunOutcome::Exhausted(checkpoint) => {
            assert!(
                checkpoint.final_value <= start_value,
                "the checkpoint rho={:?} at {:.6e} sits above the start's {start_value:.6e}",
                checkpoint.rho.to_vec(),
                checkpoint.final_value
            );
        }
        PlanRunOutcome::FirstOrderFallbackRequested(request) => {
            panic!("ARC requested a first-order fallback: {}", request.reason())
        }
        PlanRunOutcome::FixedPointContinuationRequested(request) => {
            panic!("ARC requested a fixed-point continuation: {}", request.refusal)
        }
    }
}
