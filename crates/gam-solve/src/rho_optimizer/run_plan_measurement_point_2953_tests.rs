//! #2953: an outer result's gradient is a measurement at a point, and the
//! gradient-reproducibility floor widens only when that point is the one being
//! certified.
//!
//! The incident: a one-iteration search inside a Gaussian well stopped at one ρ,
//! the dominance continuation from it stopped at a second, and the incumbent took
//! the second ρ and value while keeping the first gradient. The floor then read
//! the two gradients as a same-ρ redraw and widened the bound past the slope it
//! was judging.

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
            assert!(
                !message.contains(StationarityBoundSource::GradientReproducibility.label()),
                "a deterministic criterion has no gradient noise for the floor to measure: \
                 {message}",
            );
        }
    }
}

/// A flat criterion whose certificate-time gradient is `CERT_GRADIENT`, certified
/// at `POINT` with a run-recorded measurement of gradient `RECORDED_GRADIENT`
/// taken at `recorded_rho`. When the two measurements share a point their spread
/// is 2.5, so the floor's bound is exactly 5.0 against |Pg| = 3.
fn certify_flat_point_with_recorded_measurement(
    recorded_rho: f64,
) -> Result<OuterCriterionCertificate, EstimationError> {
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
        array![recorded_rho],
        FLAT_VALUE,
        array![RECORDED_GRADIENT],
    ));
    certify_outer_optimality(&mut obj, &config, "reproducibility floor #2953", &mut result)
}

#[test]
fn the_reproducibility_floor_widens_only_on_a_measurement_at_the_certified_rho_2953() {
    let same_point = certify_flat_point_with_recorded_measurement(0.5)
        .expect("two measurements at the certified rho whose gradients disagree widen the bound");
    assert!(same_point.certifies(), "{}", same_point.summary());
    assert_eq!(
        same_point.stationarity.rung().label,
        StationarityBoundSource::GradientReproducibility.label(),
        "{}",
        same_point.summary(),
    );
    assert_eq!(
        same_point.stationarity.bound().to_bits(),
        5.0_f64.to_bits(),
        "the widened bound is twice the spread between the two measurements: {}",
        same_point.summary(),
    );

    // One ulp away is a different point, so the spread is not a redraw.
    let neighbour = f64::from_bits(0.5_f64.to_bits() + 1);
    let refusal = certify_flat_point_with_recorded_measurement(neighbour)
        .expect_err("a measurement taken at another rho is not a redraw of this one");
    let message = refusal.to_string();
    assert!(
        message.contains("NOT STATIONARY"),
        "the refusal must be the ordinary non-stationarity one: {message}",
    );
    assert!(
        !message.contains(StationarityBoundSource::GradientReproducibility.label()),
        "the floor must not decide a point its recorded measurement was not taken at: \
         {message}",
    );
}
