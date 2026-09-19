//! #2953: a certified optimum that an evaluated state beats is declined (#2596, #2627), and
//! when nothing certifies in its place the refusal reports it instead of losing it.
//!
//! Every fixture is flat to roundoff at the neutral seed ρ = 0, so that seed certifies in zero
//! iterations at a flat top, while a search started below it stops lower without certifying.
//! The plan runner declines the flat top and continues from the state that beat it.
//! - A softplus knee over a gently curved slope: the continuation cannot certify, and the fit
//!   refuses with `DominanceUnresolved`.
//! - A Gaussian well: a budget one iteration short of the unbounded search lets the
//!   continuation certify the centre, which publishes.
//! - The well plus a concave ridge along a second coordinate: the continuation certifies a
//!   strict saddle whose escape cannot run, and the fit refuses with
//!   `IncumbentUnescapableSaddle`, or propagates the escape search's fatal failure.
//! - The well again, where the objective refuses to re-evaluate the checkpoint that beats the
//!   flat top: the flat top is still declined on the checkpoint's stored value. The fit refuses
//!   with `DominanceUnresolved`, unless a later plan attempt certifies below the checkpoint.
//! - The well where a search cannot restart from a stored state: a continuation that certifies
//!   the flat top it was started to replace, or a later plan attempt that certifies it again,
//!   declines it on the state it started from instead of publishing it.

use super::*;
use gam_problem::DominanceRefusalKind;
use ndarray::array;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Off the integer seed lattice, so no generated seed starts at the centre.
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

/// Search the well from `WELL_START` under `max_iter`, the neutral seed next in the cascade,
/// against a cache of its own so no run resumes another's checkpoint.
fn run_well(max_iter: usize, label: &str) -> Result<OuterResult, EstimationError> {
    let (_cache_dir, session) = tmp_cache_session(label);
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![-6.0], array![6.0])
        .with_initial_rho(array![WELL_START])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            risk_profile: gam_problem::SeedRiskProfile::Gaussian,
            ..Default::default()
        })
        .with_max_iter(max_iter)
        .with_cache_session(session);
    let mut objective = problem.build_objective(
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
    );
    problem.run(&mut objective, label)
}

// The refusal fixture: a softplus knee with a gentle curvature below it,
// f = −S·q + ε·q²/2 with q = W·ln(1 + e^{−(x − K)/W}) ≈ max(0, K − x). It is flat to roundoff at
// the neutral seed, and below the knee it falls with a gradient S − ε·q that barely changes along a
// search. A one-iteration search there stops low without certifying. Its run-recorded gradient and
// the certificate-time gradient then agree to about ε per unit moved, so the certificate's
// gradient-reproducibility floor cannot widen the band past the checkpoint's own gradient, and the
// checkpoint is refused on that gradient. Below the knee the minimum sits beyond the declared
// lower bound, so no search reaches a certified point there.

const KNEE: f64 = -1.0;
const KNEE_WIDTH: f64 = 0.05;
const SLOPE: f64 = 1.0;
const SLOPE_CURVATURE: f64 = 0.01;
const SLOPE_START: f64 = -2.0;

fn softplus(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn knee_depth(x: f64) -> f64 {
    KNEE_WIDTH * softplus(-(x - KNEE) / KNEE_WIDTH)
}

fn slope_value(x: f64) -> f64 {
    let q = knee_depth(x);
    -SLOPE * q + 0.5 * SLOPE_CURVATURE * q * q
}

fn slope_derivative(x: f64) -> f64 {
    logistic(-(x - KNEE) / KNEE_WIDTH) * (SLOPE - SLOPE_CURVATURE * knee_depth(x))
}

/// Search the slope from `SLOPE_START` under `max_iter`, the neutral seed next in the cascade,
/// against a cache of its own.
fn run_slope(max_iter: usize, label: &str) -> Result<OuterResult, EstimationError> {
    let (_cache_dir, session) = tmp_cache_session(label);
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![-50.0], array![6.0])
        .with_initial_rho(array![SLOPE_START])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            risk_profile: gam_problem::SeedRiskProfile::Gaussian,
            ..Default::default()
        })
        .with_max_iter(max_iter)
        .with_cache_session(session);
    let mut objective = problem.build_objective(
        (),
        |_: &mut (), rho: &Array1<f64>| Ok(slope_value(rho[0])),
        |_: &mut (), rho: &Array1<f64>| {
            Ok(OuterEval {
                cost: slope_value(rho[0]),
                gradient: array![slope_derivative(rho[0])],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem.run(&mut objective, label)
}

#[test]
fn a_declined_certified_optimum_is_reported_when_nothing_certifies_in_its_place_2953() {
    let error = run_slope(1, "dominated plateau refusal #2953").expect_err(
        "a one-iteration search cannot certify on the slope, and the flat top it beats must \
         not publish",
    );
    let EstimationError::DominatedCertifiedPlateau {
        kind,
        plateau_rho,
        plateau_value,
        incumbent_rho,
        incumbent_value,
        gap,
        band,
        continuation,
        terminal_refusal,
        ..
    } = error
    else {
        panic!("expected the typed dominated-plateau refusal, got {error}");
    };
    assert_eq!(kind, DominanceRefusalKind::DominanceUnresolved);
    // The declined optimum is the neutral seed, certified where the knee is flat to roundoff.
    assert_eq!(plateau_rho, vec![0.0]);
    assert_eq!(plateau_value.to_bits(), slope_value(0.0).to_bits());
    // The checkpoint the terminal certificate refused is on the slope, off the declared bound.
    assert!(
        incumbent_rho[0] < KNEE - 10.0 * KNEE_WIDTH && incumbent_rho[0] > -49.0,
        "the refused checkpoint must be the search on the slope, off the bound; \
         rho={incumbent_rho:?}"
    );
    assert!(
        incumbent_value < plateau_value - 0.5,
        "the refused checkpoint {incumbent_value:e} must sit well below the declined optimum \
         {plateau_value:e}"
    );
    // The decline was judged beyond the rounding envelope, by a gap of the slope's scale.
    assert!(
        band > 0.0 && gap > 0.5 && gap > band,
        "gap {gap:e} against band {band:e}"
    );
    assert!(
        continuation.starts_with("exhausted at objective")
            || continuation.starts_with("declined another certified optimum at objective"),
        "the continuation from the state that beat the optimum must have run without \
         certifying: {continuation}"
    );
    assert!(
        matches!(*terminal_refusal, EstimationError::RemlDidNotConverge { .. }),
        "the terminal certificate's own refusal must ride inside the report: {terminal_refusal}"
    );
}

#[test]
fn a_continuation_that_certifies_publishes_after_the_decline_2953() {
    // The same search without a binding budget certifies the centre, and its iteration count
    // sizes a budget that stops the first search one iteration short of it.
    let calibration = run_well(200, "dominated plateau calibration #2953")
        .expect("an unbounded search from inside the well certifies its centre");
    let unbounded_iterations = calibration.iterations;
    assert!(
        unbounded_iterations >= 2,
        "the well search must take at least two iterations for a budget to split it; took \
         {unbounded_iterations}"
    );
    let budget = unbounded_iterations - 1;
    let published = run_well(budget, "dominated plateau continuation #2953").unwrap_or_else(
        |error| {
            panic!(
                "the search continued from the state that beat the declined optimum must \
                 certify and publish: {error}"
            )
        },
    );
    assert!(
        (published.rho[0] - CENTER).abs() < 1.0e-3,
        "the published optimum must be the well's centre; rho={:?}",
        published.rho
    );
    // The first search stopped at `budget` iterations and the neutral seed took none, so a
    // larger ledger is the continuation's own work.
    assert!(
        published.iterations > budget,
        "the published fit must come from the continuation, not the budget-capped first search: \
         {} iteration(s) against a budget of {budget}",
        published.iterations
    );
}

// The strict-saddle fixture: the same well in x, and concave curvature along y. The search
// never leaves the ridge y = 0, because the gradient along y is exactly zero there, so the
// continuation certifies the well's centre under the first-order screening certificate and
// the terminal certificate refuses that point as a strict saddle. Its escape descends along
// y, and the objective refuses every evaluation once an off-ridge evaluation has been
// followed by a reset, which first happens when the certify loop starts the escape's search.

const RIDGE_CURVATURE: f64 = 4.0;
const ARMED_MARKER: &str = "the #2953 ridge fixture refuses evaluation after the saddle escape";

fn well_curvature(x: f64) -> f64 {
    let u = x - CENTER;
    -well_value(x) * (1.0 / (WIDTH * WIDTH) - u * u / WIDTH.powi(4))
}

fn ridge_value(rho: &Array1<f64>) -> f64 {
    well_value(rho[0]) - 0.5 * RIDGE_CURVATURE * rho[1] * rho[1]
}

/// What the ridge objective raises once armed.
#[derive(Clone, Copy)]
enum ArmedRefusal {
    /// A refusal of the trial point, which a search walks away from.
    TrialPoint,
    /// A failure of the evaluation itself, which stops the fit.
    Fatal,
}

struct RidgeState {
    refusal: Option<ArmedRefusal>,
    off_ridge: bool,
    armed: bool,
    armed_refusals: Arc<AtomicUsize>,
}

fn ridge_refusal(state: &mut RidgeState, rho: &Array1<f64>) -> Result<(), EstimationError> {
    if state.armed
        && let Some(refusal) = state.refusal
    {
        state.armed_refusals.fetch_add(1, Ordering::Relaxed);
        return Err(match refusal {
            ArmedRefusal::TrialPoint => EstimationError::TrialPointRefused {
                reason: ARMED_MARKER.to_string(),
            },
            ArmedRefusal::Fatal => EstimationError::fatal_outer_evaluation(
                "ridge fixture",
                EstimationError::InvalidInput(ARMED_MARKER.to_string()),
            ),
        });
    }
    if rho[1] != 0.0 {
        state.off_ridge = true;
    }
    Ok(())
}

/// The ridge problem from `WELL_START` on the ridge under `max_iter`: a gradient-only search
/// over a declared analytic Hessian, which the terminal certificate measures.
fn ridge_problem(max_iter: usize, label: &str) -> (tempfile::TempDir, OuterProblem) {
    let (cache_dir, session) = tmp_cache_session(label);
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_prefer_gradient_only(true)
        .with_fallback_policy(FallbackPolicy::Disabled)
        .with_bounds(array![-6.0, -6.0], array![6.0, 6.0])
        .with_initial_rho(array![WELL_START, 0.0])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            risk_profile: gam_problem::SeedRiskProfile::Gaussian,
            ..Default::default()
        })
        .with_max_iter(max_iter)
        .with_cache_session(session);
    (cache_dir, problem)
}

macro_rules! ridge_objective {
    ($problem:expr, $refusal:expr, $armed_refusals:expr) => {
        $problem.build_objective(
            RidgeState {
                refusal: $refusal,
                off_ridge: false,
                armed: false,
                armed_refusals: $armed_refusals,
            },
            |state: &mut RidgeState, rho: &Array1<f64>| {
                ridge_refusal(state, rho)?;
                Ok(ridge_value(rho))
            },
            |state: &mut RidgeState, rho: &Array1<f64>| {
                ridge_refusal(state, rho)?;
                Ok(OuterEval {
                    cost: ridge_value(rho),
                    gradient: array![well_derivative(rho[0]), -RIDGE_CURVATURE * rho[1]],
                    hessian: HessianValue::Dense(array![
                        [well_curvature(rho[0]), 0.0],
                        [0.0, -RIDGE_CURVATURE]
                    ]),
                    inner_beta_hint: None,
                })
            },
            Some(|state: &mut RidgeState| {
                if state.off_ridge {
                    state.armed = true;
                }
            }),
            None::<fn(&mut RidgeState, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
    };
}

/// The ridge search from `WELL_START` without a binding budget, judged by the screening
/// certificate alone: the certified result, or `None` when it does not certify. The callers
/// check it, since a helper outside a `#[test]` does not panic.
fn ridge_screening_certified() -> Result<Option<OuterResult>, EstimationError> {
    let (_cache_dir, problem) = ridge_problem(200, "ridge calibration #2953");
    let mut objective = ridge_objective!(problem, None, Arc::new(AtomicUsize::new(0)));
    let config = problem.config();
    let capability = objective.capability();
    let the_plan = plan(&capability);
    Ok(
        match run_outer_with_plan(
            &mut objective,
            &config,
            "ridge calibration #2953",
            &capability,
            &the_plan,
            false,
        )? {
            PlanRunOutcome::Converged(result) => Some(result),
            _ => None,
        },
    )
}

#[test]
fn a_declined_optimum_beaten_by_an_unescapable_strict_saddle_is_reported_by_kind_2953() {
    let calibration = ridge_screening_certified()
        .expect("the ridge search from inside the well must run")
        .expect(
            "an unbounded ridge search from inside the well must certify its centre under the \
             screening certificate",
        );
    assert!(
        (calibration.rho[0] - CENTER).abs() < 1.0e-3 && calibration.rho[1] == 0.0,
        "the unbounded ridge search must certify the saddle at the well's centre; rho={:?}",
        calibration.rho
    );
    let unbounded_iterations = calibration.iterations;
    assert!(
        unbounded_iterations >= 2,
        "the ridge search must take at least two iterations for a budget to split it; took \
         {unbounded_iterations}"
    );
    let armed_refusals = Arc::new(AtomicUsize::new(0));
    let (_cache_dir, problem) =
        ridge_problem(unbounded_iterations - 1, "ridge saddle refusal #2953");
    let mut objective = ridge_objective!(
        problem,
        Some(ArmedRefusal::TrialPoint),
        Arc::clone(&armed_refusals)
    );
    let error = problem
        .run(&mut objective, "ridge saddle refusal #2953")
        .expect_err("a strict saddle whose escape cannot run must not publish");
    let EstimationError::DominatedCertifiedPlateau {
        kind,
        plateau_rho,
        incumbent_rho,
        gap,
        band,
        continuation,
        terminal_refusal,
        ..
    } = error
    else {
        panic!("expected the typed dominated-plateau refusal, got {error}");
    };
    assert_eq!(kind, DominanceRefusalKind::IncumbentUnescapableSaddle);
    assert_eq!(plateau_rho, vec![0.0, 0.0]);
    assert!(
        (incumbent_rho[0] - CENTER).abs() < 1.0e-3 && incumbent_rho[1] == 0.0,
        "the refused checkpoint must be the saddle at the well's centre on the ridge; \
         rho={incumbent_rho:?}"
    );
    assert!(
        band > 0.0 && gap > DEPTH / 2.0 && gap > band,
        "gap {gap:e} against band {band:e}"
    );
    assert!(
        continuation.starts_with("certified at objective"),
        "the continuation must have certified the saddle under the screening certificate: \
         {continuation}"
    );
    assert!(
        armed_refusals.load(Ordering::Relaxed) > 0,
        "the saddle escape must have armed the objective, so its search could not run"
    );
    assert!(
        terminal_refusal
            .to_string()
            .contains("SaddleEscape reseed could not run"),
        "the terminal refusal must carry why the escape was not taken: {terminal_refusal}"
    );
}

#[test]
fn a_fatal_failure_of_the_saddle_escape_search_propagates_as_it_is_2953() {
    let calibration = ridge_screening_certified()
        .expect("the ridge search from inside the well must run")
        .expect(
            "an unbounded ridge search from inside the well must certify its centre under the \
             screening certificate",
        );
    assert!(
        (calibration.rho[0] - CENTER).abs() < 1.0e-3 && calibration.rho[1] == 0.0,
        "the unbounded ridge search must certify the saddle at the well's centre; rho={:?}",
        calibration.rho
    );
    let unbounded_iterations = calibration.iterations;
    assert!(
        unbounded_iterations >= 2,
        "the ridge search must take at least two iterations for a budget to split it; took \
         {unbounded_iterations}"
    );
    let armed_refusals = Arc::new(AtomicUsize::new(0));
    let (_cache_dir, problem) =
        ridge_problem(unbounded_iterations - 1, "ridge fatal escape #2953");
    let mut objective =
        ridge_objective!(problem, Some(ArmedRefusal::Fatal), Arc::clone(&armed_refusals));
    let error = problem
        .run(&mut objective, "ridge fatal escape #2953")
        .expect_err("a fatal evaluation failure must stop the fit");
    assert!(
        error.is_fatal_outer_evaluation() && error.to_string().contains(ARMED_MARKER),
        "the escape search's fatal failure must be the refusal, not an older certification \
         refusal: {error}"
    );
    assert!(
        armed_refusals.load(Ordering::Relaxed) > 0,
        "the saddle escape must have armed the objective before the fatal failure"
    );
}

// The re-entry fixture: the checkpoint that beats the declined optimum cannot be re-evaluated
// at its own ρ. The objective refuses, as an infeasible trial, a point it evaluated before its
// latest reset, as an objective does whose inner state at a stored checkpoint a cold solve
// cannot re-enter. The plan runner re-evaluates the checkpoint from a reset before it may
// outrank the certified winner, so that re-evaluation is the first refusal. The well is
// searched from inside its convex core, where a Newton-like step on the declared curvature
// lands near the centre.

const REENTRY_START: f64 = -3.7;
const REENTRY_MARKER: &str =
    "the #2953 re-entry fixture refuses a point it evaluated before its last reset";

struct ReentryState {
    /// Points evaluated before the latest reset.
    earlier: Vec<Array1<f64>>,
    /// Points evaluated since the latest reset.
    since_reset: Vec<Array1<f64>>,
    /// How many more re-entries it refuses.
    refusals_left: usize,
    refusals: Arc<AtomicUsize>,
}

fn reentry_refusal(state: &mut ReentryState, rho: &Array1<f64>) -> Result<(), EstimationError> {
    let reentry = state.earlier.iter().any(|point| {
        point.len() == rho.len()
            && point
                .iter()
                .zip(rho.iter())
                .all(|(earlier, now)| earlier.to_bits() == now.to_bits())
    });
    if reentry && state.refusals_left > 0 {
        state.refusals_left -= 1;
        state.refusals.fetch_add(1, Ordering::Relaxed);
        return Err(EstimationError::TrialPointRefused {
            reason: REENTRY_MARKER.to_string(),
        });
    }
    state.since_reset.push(rho.clone());
    Ok(())
}

/// The well from `REENTRY_START` under `max_iter`, the neutral seed next in the cascade, with
/// its analytic curvature declared. `prefer_gradient_only` makes the first attempt BFGS, and
/// `fallback` decides whether ARC on the declared curvature follows it (#2898).
fn reentry_problem(
    max_iter: usize,
    prefer_gradient_only: bool,
    fallback: FallbackPolicy,
    label: &str,
) -> (tempfile::TempDir, OuterProblem) {
    let (cache_dir, session) = tmp_cache_session(label);
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_prefer_gradient_only(prefer_gradient_only)
        .with_fallback_policy(fallback)
        .with_bounds(array![-6.0], array![6.0])
        .with_initial_rho(array![REENTRY_START])
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            risk_profile: gam_problem::SeedRiskProfile::Gaussian,
            ..Default::default()
        })
        .with_max_iter(max_iter)
        .with_cache_session(session);
    (cache_dir, problem)
}

macro_rules! reentry_objective {
    ($problem:expr, $refusals_left:expr, $refusals:expr) => {
        $problem.build_objective(
            ReentryState {
                earlier: Vec::new(),
                since_reset: Vec::new(),
                refusals_left: $refusals_left,
                refusals: $refusals,
            },
            |state: &mut ReentryState, rho: &Array1<f64>| {
                reentry_refusal(state, rho)?;
                Ok(well_value(rho[0]))
            },
            |state: &mut ReentryState, rho: &Array1<f64>| {
                reentry_refusal(state, rho)?;
                Ok(OuterEval {
                    cost: well_value(rho[0]),
                    gradient: array![well_derivative(rho[0])],
                    hessian: HessianValue::Dense(array![[well_curvature(rho[0])]]),
                    inner_beta_hint: None,
                })
            },
            Some(|state: &mut ReentryState| {
                let since_reset = std::mem::take(&mut state.since_reset);
                state.earlier.extend(since_reset);
            }),
            None::<fn(&mut ReentryState, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
    };
}

/// The iterations an unbounded single-attempt search of the well takes to certify with nothing
/// refused: BFGS when `prefer_gradient_only`, ARC on the declared curvature otherwise.
fn reentry_calibration(prefer_gradient_only: bool) -> Result<usize, EstimationError> {
    let label =
        format!("re-entry calibration #2953 (prefer_gradient_only={prefer_gradient_only})");
    let (_cache_dir, problem) =
        reentry_problem(200, prefer_gradient_only, FallbackPolicy::Disabled, &label);
    let mut objective = reentry_objective!(problem, 0, Arc::new(AtomicUsize::new(0)));
    Ok(problem.run(&mut objective, &label)?.iterations)
}

#[test]
fn a_checkpoint_that_cannot_be_re_evaluated_still_declines_the_optimum_it_beats_2953() {
    let bfgs_iterations = reentry_calibration(true)
        .expect("an unbounded BFGS search of the well from inside its core certifies its centre");
    assert!(
        bfgs_iterations >= 2,
        "the BFGS search must take at least two iterations for a budget to split it; took \
         {bfgs_iterations}"
    );
    let refusals = Arc::new(AtomicUsize::new(0));
    let (_cache_dir, problem) = reentry_problem(
        bfgs_iterations - 1,
        true,
        FallbackPolicy::Disabled,
        "re-entry refusal #2953",
    );
    let mut objective = reentry_objective!(problem, usize::MAX, Arc::clone(&refusals));
    let error = problem
        .run(&mut objective, "re-entry refusal #2953")
        .expect_err(
            "a certified optimum that a stored checkpoint beats must not publish because the \
             checkpoint cannot be re-evaluated",
        );
    let EstimationError::DominatedCertifiedPlateau {
        kind,
        plateau_rho,
        plateau_value,
        incumbent_rho,
        incumbent_value,
        gap,
        band,
        continuation,
        terminal_refusal,
        ..
    } = error
    else {
        panic!("expected the typed dominated-plateau refusal, got {error}");
    };
    assert_eq!(kind, DominanceRefusalKind::DominanceUnresolved);
    assert_eq!(plateau_rho, vec![0.0]);
    assert_eq!(plateau_value.to_bits(), well_value(0.0).to_bits());
    // The checkpoint is where the capped search stopped, inside the well, at its stored value:
    // no re-evaluation replaced it.
    assert!(
        (incumbent_rho[0] - CENTER).abs() < WIDTH,
        "the refused checkpoint must be the capped search inside the well; rho={incumbent_rho:?}"
    );
    assert_eq!(incumbent_value.to_bits(), well_value(incumbent_rho[0]).to_bits());
    assert_eq!(gap.to_bits(), (plateau_value - incumbent_value).to_bits());
    assert!(
        band > 0.0 && gap > DEPTH / 2.0 && gap > band,
        "gap {gap:e} against band {band:e}"
    );
    assert!(
        continuation.contains("re-evaluating the checkpoint at its own rho was refused")
            && continuation.contains(REENTRY_MARKER),
        "the continuation must say that no search could start from the refused checkpoint: \
         {continuation}"
    );
    assert!(
        terminal_refusal.is_trial_point_infeasible(),
        "the terminal certificate's installation at the checkpoint is refused the same way: \
         {terminal_refusal}"
    );
    assert!(
        refusals.load(Ordering::Relaxed) >= 2,
        "both the dominance re-evaluation and the terminal installation must have been refused; \
         refused {}",
        refusals.load(Ordering::Relaxed)
    );
}

#[test]
fn a_later_attempt_that_certifies_below_the_refused_checkpoint_publishes_2953() {
    let bfgs_iterations = reentry_calibration(true)
        .expect("an unbounded BFGS search of the well from inside its core certifies its centre");
    let arc_iterations = reentry_calibration(false)
        .expect("an unbounded ARC search of the well from inside its core certifies its centre");
    // Under a budget of ARC's own iteration count the BFGS attempt stops short of
    // certifying, and the ARC attempt that follows it certifies.
    assert!(
        arc_iterations < bfgs_iterations,
        "ARC on the declared curvature must certify in fewer iterations than BFGS for one \
         budget to split them: ARC {arc_iterations}, BFGS {bfgs_iterations}"
    );
    let refusals = Arc::new(AtomicUsize::new(0));
    let (_cache_dir, problem) = reentry_problem(
        arc_iterations,
        true,
        FallbackPolicy::Automatic,
        "re-entry continuation #2953",
    );
    let mut objective = reentry_objective!(problem, 1, Arc::clone(&refusals));
    let published = problem
        .run(&mut objective, "re-entry continuation #2953")
        .unwrap_or_else(|error| {
            panic!(
                "the ARC attempt certifies the well's centre, below the refused checkpoint, and \
                 must publish: {error}"
            )
        });
    assert!(
        (published.rho[0] - CENTER).abs() < 1.0e-3,
        "the published optimum must be the well's centre; rho={:?}",
        published.rho
    );
    assert_eq!(
        refusals.load(Ordering::Relaxed),
        1,
        "the BFGS attempt's dominance re-evaluation must have been the one refusal"
    );
    let Some(record) = published.dominated_plateau.as_ref() else {
        panic!("the BFGS attempt's declined optimum must ride on the published result");
    };
    assert_eq!(record.plateau_rho.to_vec(), vec![0.0]);
    assert!(
        matches!(
            &record.continuation,
            DominanceContinuationStop::Failed { error }
                if error.contains("re-evaluating the checkpoint at its own rho was refused")
        ),
        "the BFGS attempt must have declined on the checkpoint it could not re-evaluate: {}",
        record.continuation
    );
}

// The continuation fixture: the re-entry well, where a search cannot restart from a stored state,
// under a shallow bowl centred at the neutral seed, so the flat top is a genuine local minimum that
// every certificate accepts. The objective refuses once, as an infeasible trial, a derivative
// evaluation at a point it evaluated before its latest reset; value-only evaluations always price.
// So the dominance re-evaluation of the checkpoint prices it, and the continuation's first step from
// that checkpoint is refused. The continuation's cascade then certifies the bowl's minimum: the very
// optimum it was started to replace. Below `SECOND_ORDER_EDGE` every second-order evaluation is
// refused as well, so an ARC attempt cannot search the well either.

/// The bowl's curvature: small against the well's, so the well's centre stays far below the bowl's
/// minimum, and positive, so the bowl's minimum certifies at every order.
const BOWL_CURVATURE: f64 = 1.0e-3;
const SECOND_ORDER_EDGE: f64 = -1.0;
const SECOND_ORDER_MARKER: &str =
    "the #2953 continuation fixture refuses a second-order evaluation inside the well";

fn bowl_value(x: f64) -> f64 {
    well_value(x) + 0.5 * BOWL_CURVATURE * x * x
}

fn bowl_derivative(x: f64) -> f64 {
    well_derivative(x) + BOWL_CURVATURE * x
}

fn bowl_curvature(x: f64) -> f64 {
    well_curvature(x) + BOWL_CURVATURE
}

struct ContinuationState {
    reentry: ReentryState,
    refuse_second_order: bool,
    second_order_refusals: Arc<AtomicUsize>,
}

fn second_order_refusal(state: &ContinuationState, rho: &Array1<f64>) -> Result<(), EstimationError> {
    if state.refuse_second_order && rho[0] < SECOND_ORDER_EDGE {
        state.second_order_refusals.fetch_add(1, Ordering::Relaxed);
        return Err(EstimationError::TrialPointRefused {
            reason: SECOND_ORDER_MARKER.to_string(),
        });
    }
    Ok(())
}

fn bowl_eval(rho: &Array1<f64>, hessian: HessianValue) -> OuterEval {
    OuterEval {
        cost: bowl_value(rho[0]),
        gradient: array![bowl_derivative(rho[0])],
        hessian,
        inner_beta_hint: None,
    }
}

fn well_eval(rho: &Array1<f64>, hessian: HessianValue) -> OuterEval {
    OuterEval {
        cost: well_value(rho[0]),
        gradient: array![well_derivative(rho[0])],
        hessian,
        inner_beta_hint: None,
    }
}

macro_rules! continuation_objective {
    ($problem:expr, $refusals_left:expr, $refuse_second_order:expr, $reentry_refusals:expr,
     $second_order_refusals:expr) => {
        $problem.build_objective_with_eval_order(
            ContinuationState {
                reentry: ReentryState {
                    earlier: Vec::new(),
                    since_reset: Vec::new(),
                    refusals_left: $refusals_left,
                    refusals: $reentry_refusals,
                },
                refuse_second_order: $refuse_second_order,
                second_order_refusals: $second_order_refusals,
            },
            |state: &mut ContinuationState, rho: &Array1<f64>| {
                state.reentry.since_reset.push(rho.clone());
                Ok(bowl_value(rho[0]))
            },
            |state: &mut ContinuationState, rho: &Array1<f64>| {
                second_order_refusal(state, rho)?;
                reentry_refusal(&mut state.reentry, rho)?;
                Ok(bowl_eval(rho, HessianValue::Dense(array![[bowl_curvature(rho[0])]])))
            },
            |state: &mut ContinuationState, rho: &Array1<f64>, order: OuterEvalOrder| {
                match order {
                    OuterEvalOrder::Value => state.reentry.since_reset.push(rho.clone()),
                    OuterEvalOrder::ValueAndGradient => reentry_refusal(&mut state.reentry, rho)?,
                    OuterEvalOrder::ValueGradientHessian => {
                        second_order_refusal(state, rho)?;
                        reentry_refusal(&mut state.reentry, rho)?;
                    }
                }
                Ok(bowl_eval(
                    rho,
                    match order {
                        OuterEvalOrder::ValueGradientHessian => {
                            HessianValue::Dense(array![[bowl_curvature(rho[0])]])
                        }
                        OuterEvalOrder::Value | OuterEvalOrder::ValueAndGradient => {
                            HessianValue::Unavailable
                        }
                    },
                ))
            },
            Some(|state: &mut ContinuationState| {
                let since_reset = std::mem::take(&mut state.reentry.since_reset);
                state.reentry.earlier.extend(since_reset);
            }),
            None::<fn(&mut ContinuationState, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        )
    };
}

/// The iterations an unbounded BFGS search of the bowl-and-well takes to certify the well's centre
/// from inside its core, with nothing refused.
fn continuation_calibration() -> Result<usize, EstimationError> {
    let label = "continuation calibration #2953";
    let (_cache_dir, problem) = reentry_problem(200, true, FallbackPolicy::Disabled, label);
    let mut objective = continuation_objective!(
        problem,
        0,
        false,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0))
    );
    Ok(problem.run(&mut objective, label)?.iterations)
}

struct ContinuationRun {
    outcome: Result<OuterResult, EstimationError>,
    reentry_refusals: usize,
    second_order_refusals: usize,
}

/// The continuation fixture under a BFGS budget one iteration short of its unbounded search.
fn run_continuation_fixture(
    fallback: FallbackPolicy,
    label: &str,
) -> Result<ContinuationRun, EstimationError> {
    let bfgs_iterations = continuation_calibration()?;
    let reentry_refusals = Arc::new(AtomicUsize::new(0));
    let second_order_refusals = Arc::new(AtomicUsize::new(0));
    let (_cache_dir, problem) =
        reentry_problem(bfgs_iterations.saturating_sub(1).max(1), true, fallback, label);
    let mut objective = continuation_objective!(
        problem,
        1,
        true,
        Arc::clone(&reentry_refusals),
        Arc::clone(&second_order_refusals)
    );
    let outcome = problem.run(&mut objective, label);
    Ok(ContinuationRun {
        outcome,
        reentry_refusals: reentry_refusals.load(Ordering::Relaxed),
        second_order_refusals: second_order_refusals.load(Ordering::Relaxed),
    })
}

#[test]
fn a_continuation_cannot_publish_the_optimum_it_was_started_to_replace_2953() {
    let bfgs_iterations = continuation_calibration()
        .expect("an unbounded BFGS search of the well from inside its core certifies its centre");
    assert!(
        bfgs_iterations >= 2,
        "the BFGS search must take at least two iterations for a budget to split it; took \
         {bfgs_iterations}"
    );
    let run = run_continuation_fixture(FallbackPolicy::Disabled, "continuation dominance #2953")
        .expect("the continuation fixture's calibration certifies the well's centre");
    let error = run.outcome.expect_err(
        "the continuation re-certified the bowl's minimum it was started to replace, and that \
         minimum must not publish over the state the continuation started from",
    );
    let EstimationError::DominatedCertifiedPlateau {
        plateau_rho,
        incumbent_rho,
        continuation,
        ..
    } = error
    else {
        panic!("expected the typed dominated-plateau refusal, got {error}");
    };
    assert!(
        plateau_rho[0] > SECOND_ORDER_EDGE,
        "the declined optimum must be the bowl's minimum, outside the well; rho={plateau_rho:?}"
    );
    assert!(
        (incumbent_rho[0] - CENTER).abs() < WIDTH,
        "the checkpoint that beat the bowl's minimum must be the capped search's, inside the \
         well; rho={incumbent_rho:?}"
    );
    assert!(
        continuation.starts_with("declined another certified optimum at objective"),
        "the continuation must have declined the bowl's minimum it certified again: {continuation}"
    );
    assert_eq!(
        run.reentry_refusals, 1,
        "the continuation's first step from the checkpoint must have been the one re-entry refusal"
    );
}

#[test]
fn a_later_attempt_cannot_publish_the_optimum_an_earlier_attempt_declined_2953() {
    let bfgs_iterations = continuation_calibration()
        .expect("an unbounded BFGS search of the well from inside its core certifies its centre");
    assert!(
        bfgs_iterations >= 2,
        "the BFGS search must take at least two iterations for a budget to split it; took \
         {bfgs_iterations}"
    );
    let run = run_continuation_fixture(FallbackPolicy::Automatic, "cross-attempt dominance #2953")
        .expect("the continuation fixture's calibration certifies the well's centre");
    let error = run.outcome.expect_err(
        "the bowl's minimum that the BFGS attempt declined for a state in the well must not \
         publish when the ARC attempt certifies it again",
    );
    let EstimationError::DominatedCertifiedPlateau {
        plateau_rho,
        incumbent_rho,
        gap,
        band,
        ..
    } = error
    else {
        panic!("expected the typed dominated-plateau refusal, got {error}");
    };
    assert!(
        plateau_rho[0] > SECOND_ORDER_EDGE,
        "the declined optimum must be the bowl's minimum, outside the well; rho={plateau_rho:?}"
    );
    assert!(
        (incumbent_rho[0] - CENTER).abs() < WIDTH,
        "the checkpoint that beat the bowl's minimum must be the BFGS attempt's, inside the well; \
         rho={incumbent_rho:?}"
    );
    assert!(
        band > 0.0 && gap > DEPTH / 2.0 && gap > band,
        "gap {gap:e} against band {band:e}"
    );
    assert_eq!(
        run.reentry_refusals, 1,
        "the BFGS continuation's first step must have been the one re-entry refusal"
    );
    assert!(
        run.second_order_refusals > 0,
        "the ARC attempt's second-order evaluations inside the well must have been refused"
    );
}

/// A carried checkpoint is compared at the current criterion's value, not at the value an earlier
/// attempt stored. The carried state claims a stored value twice the well's depth below the flat
/// top, as a criterion another attempt priced differently would. Under this attempt's criterion it
/// is the flat top, well above the centre this attempt certifies, so the centre publishes. The
/// objective is evaluated at the carried ρ exactly once: the re-evaluation that priced it.
#[test]
fn a_carried_checkpoint_is_repriced_under_the_current_criterion_before_it_declines_anything_2953() {
    let label = "carried checkpoint repriced #2953";
    let (_cache_dir, problem) = reentry_problem(200, true, FallbackPolicy::Disabled, label);
    let carried_rho = array![0.0];
    let evaluations_at_carried_rho = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&evaluations_at_carried_rho);
    let count_at = move |rho: &Array1<f64>| {
        if rho[0] == 0.0 {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    };
    let count_at_eval = count_at.clone();
    let mut objective = problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| {
            count_at(rho);
            Ok(well_value(rho[0]))
        },
        move |_: &mut (), rho: &Array1<f64>| {
            count_at_eval(rho);
            Ok(well_eval(rho, HessianValue::Dense(array![[well_curvature(rho[0])]])))
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let capability = objective.capability();
    let the_plan = plan(&capability);
    let mut config = problem.config();
    let stale_value = 2.0 * well_value(CENTER);
    config.carried_checkpoint =
        Some(OuterResult::new(carried_rho.clone(), stale_value, 0, false, the_plan));
    let outcome = run_outer_with_plan(&mut objective, &config, label, &capability, &the_plan, true)
        .expect("the well search from inside its core must run");
    let PlanRunOutcome::Converged(result) = outcome else {
        panic!(
            "the certified centre must publish: under this criterion the carried state is the flat \
             top, not the stored value {stale_value:e}"
        );
    };
    assert!(
        (result.rho[0] - CENTER).abs() < 1.0e-3,
        "the published optimum must be the well's centre; rho={:?}",
        result.rho
    );
    assert_eq!(
        evaluations_at_carried_rho.load(Ordering::Relaxed),
        1,
        "the carried state must be re-evaluated at its own rho exactly once before it is compared"
    );
}
