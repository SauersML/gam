//! #2953: an outer result's gradient is a measurement at a point.
//!
//! The incident: a one-iteration search inside a Gaussian well stopped at one ρ,
//! the dominance continuation from it stopped at a second, and the incumbent took
//! the second ρ and value while keeping the first gradient.
//!
//! #3531: a second measurement at the same ρ is not a noise sample either. The
//! solver's recorded measurement comes from a capped inner solve, so its gradient
//! is off at first order in the inner truncation while its value is off only at
//! second order, and the certificate judges `|Pg|` on its own evaluation alone.

use super::*;
use ndarray::array;

/// Off every start the fixtures search from.
const CENTER: f64 = -3.5;
const WIDTH: f64 = 0.5;
const DEPTH: f64 = 10.0;
/// Inside the well's convex core and off its centre.
const WELL_START: f64 = -3.9;

fn well_value(x: f64) -> f64 {
    -DEPTH * (-(x - CENTER).powi(2) / (2.0 * WIDTH * WIDTH)).exp()
}

fn well_derivative(x: f64) -> f64 {
    -well_value(x) * (x - CENTER) / (WIDTH * WIDTH)
}

/// The well searched from `WELL_START` under a one-iteration budget. The search
/// stops inside the well without certifying, and a later search from the flat top
/// at ρ = 0, which certifies there, carries that stop as its checkpoint, as the
/// next plan attempt does, and declines the flat top.
fn well_problem() -> OuterProblem {
    OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![-6.0], array![6.0])
        .with_initial_rho(array![WELL_START])
        .with_max_iter(1)
}

fn well_objective() -> impl OuterObjective {
    well_problem().build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(well_value(rho[0])),
        |_: &mut (), rho: &Array1<f64>| {
            Ok(OuterEval {
                cost: well_value(rho[0]),
                gradient: array![well_derivative(rho[0])],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    )
}

/// The resume checkpoint of the declined attempt, with the continuation from it
/// enabled or not.
fn dominated_well_incumbent(
    obj: &mut dyn OuterObjective,
    config: &OuterConfig,
    continue_from_incumbent: bool,
    context: &str,
) -> Result<OuterResult, String> {
    let cap = obj.capability();
    let the_plan = plan(&cap);
    let stop = match run_outer_with_plan(obj, config, context, &cap, &the_plan, false) {
        Ok(PlanRunOutcome::Exhausted(stop)) => stop,
        Ok(_) => {
            return Err(format!(
                "{context}: a one-iteration search inside the well must stop without certifying"
            ));
        }
        Err(error) => return Err(format!("{context}: the capped well search failed: {error}")),
    };
    let mut flat_top = config.clone();
    flat_top.initial_rho = Some(array![0.0]);
    flat_top.carried_checkpoint = Some(carried_checkpoint_of(&stop));
    match run_outer_with_plan(obj, &flat_top, context, &cap, &the_plan, continue_from_incumbent) {
        Ok(PlanRunOutcome::DominatedPlateau(dominated)) => Ok(dominated.incumbent),
        Ok(_) => Err(format!(
            "{context}: a one-iteration search inside the well cannot certify, so the flat \
             top it beats must be declined"
        )),
        Err(error) => Err(format!("{context}: the well attempt failed: {error}")),
    }
}

#[test]
fn the_dominance_incumbent_is_measured_at_its_own_rho_2953() {
    let problem = well_problem();
    let config = problem.config();
    let mut first = well_objective();
    let first_stop = dominated_well_incumbent(&mut first, &config, false, "first stop #2953")
        .expect("the declined attempt returns its resume checkpoint");
    let mut continued = well_objective();
    let incumbent = dominated_well_incumbent(&mut continued, &config, true, "continuation #2953")
        .expect("the declined attempt returns its resume checkpoint");

    // Without this the continuation never replaced the incumbent, and the
    // measurement below would be the first search's own.
    assert!(
        incumbent.final_value < first_stop.final_value,
        "the continuation must move the incumbent below the first search's stop: \
         rho {:?} cost {:e} against rho {:?} cost {:e}",
        incumbent.rho,
        incumbent.final_value,
        first_stop.rho,
        first_stop.final_value,
    );
    let measurement = incumbent
        .final_measurement
        .as_ref()
        .expect("a gradient-based search records its terminal measurement");
    assert!(
        measurement.is_at(&incumbent.rho),
        "the incumbent at rho {:?} carries a measurement taken at rho {:?}",
        incumbent.rho,
        measurement.rho(),
    );
    assert_eq!(
        measurement.gradient()[0].to_bits(),
        well_derivative(incumbent.rho[0]).to_bits(),
        "the incumbent's gradient {:e} must be the slope at its own rho {:?}, {:e}",
        measurement.gradient()[0],
        incumbent.rho,
        well_derivative(incumbent.rho[0]),
    );
    assert_eq!(
        measurement.value().to_bits(),
        well_value(incumbent.rho[0]).to_bits(),
        "the incumbent's measured value must be the criterion at its own rho {:?}",
        incumbent.rho,
    );
}

#[test]
fn the_dominance_incumbent_does_not_certify_on_the_slope_between_two_points_2953() {
    let problem = well_problem();
    let config = problem.config();
    let mut obj = well_objective();
    let mut incumbent = dominated_well_incumbent(&mut obj, &config, true, "incumbent mint #2953")
        .expect("the declined attempt returns its resume checkpoint");
    let slope = well_derivative(incumbent.rho[0]);
    assert!(
        slope.abs() > 0.1,
        "the fixture needs an incumbent with real slope; rho {:?} has V' = {slope:e}",
        incumbent.rho,
    );
    match certify_outer_optimality(&mut obj, &config, "incumbent mint #2953", &mut incumbent) {
        Ok(certificate) => panic!(
            "a point with slope {slope:e} at rho {:?} was certified: {}",
            incumbent.rho,
            certificate.summary(),
        ),
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("NOT STATIONARY"),
                "the refusal must be the ordinary non-stationarity one: {message}",
            );
        }
    }
}

/// A flat criterion whose certificate-time gradient is `CERT_GRADIENT` = 3,
/// certified at its own ρ while the result carries the solver's measurement at
/// that same ρ with a different gradient and the same value, as a capped inner
/// solve returns: first-order error in the gradient, none visible in the value.
/// The two gradients differ by 2.5, so the retired floor bounded `|Pg|` by 5.0
/// and minted the point.
#[test]
fn a_same_rho_gradient_disagreement_does_not_widen_the_stationarity_bound_3531() {
    const POINT: f64 = 0.5;
    const FLAT_VALUE: f64 = 1.0;
    const CERT_GRADIENT: f64 = 3.0;
    const RECORDED_GRADIENT: f64 = 0.5;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable);
    let config = problem.config();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(FLAT_VALUE),
        |_: &mut (), _: &Array1<f64>| {
            Ok(OuterEval {
                cost: FLAT_VALUE,
                gradient: array![CERT_GRADIENT],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut result = OuterResult::new(
        array![POINT],
        FLAT_VALUE,
        1,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    result.final_measurement = Some(OuterFirstOrderMeasurement::new(
        array![POINT],
        FLAT_VALUE,
        array![RECORDED_GRADIENT],
    ));
    for fidelity in [CertificationFidelity::Screening, CertificationFidelity::Mint] {
        match certify_outer_optimality_with_fidelity(
            &mut obj,
            &config,
            "same-rho gradient disagreement #3531",
            &mut result,
            fidelity,
        ) {
            Ok(certificate) => panic!(
                "{fidelity:?}: |Pg| = {CERT_GRADIENT} on a flat criterion was certified on the \
                 spread between two measurements: {}",
                certificate.summary(),
            ),
            Err(error) => {
                let message = error.to_string();
                assert!(
                    message.contains("NOT STATIONARY"),
                    "{fidelity:?}: the refusal must be the ordinary non-stationarity one: \
                     {message}",
                );
            }
        }
    }
}
