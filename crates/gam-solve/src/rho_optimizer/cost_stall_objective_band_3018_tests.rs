// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): the cost-stall guard judges a decrease against the objective bands
// of the two evaluations it compares, not a relative floor (#3018). Scope comes
// from the parent via `use super::*`.
//
// The defect these pin. The guard counted an accepted step as "no
// improvement" when it bought at most `rel_tol·(1 + |V|)`, with `rel_tol`
// floored at `COST_STALL_REL_TOL_FLOOR = 1e-7`. Neither number is a property
// of the evaluations. At `|V| = 1e5` the floor is `1e-2`, so a run descending
// by `2.5e-3` per step, a decrease its evaluations resolve eleven orders of
// magnitude over, filled a window every `k` steps and the dense ARC route
// stopped it with `OUTER_ARC_UNPROGRESSING_STALL` at `|g| = 1`. The reverse
// held too: a decrease inside the inner mode's residual error cleared the
// floor and reset the window, so the guard read noise as progress.
//
// Each computed value sits within its own `band_f` of the exact criterion, so
// a decrease is resolved exactly when it exceeds `band_f(V_k) + band_f(V_{k+1})`.

use super::*;
use crate::estimate::outer_eval_capture::{
    CertificateCriterion, CertificateEvidence, InnerFactorCondition, InnerResidualCharge,
    InnerResidualSource,
};
use crate::rho_optimizer::decrement_bands::{DecrementVerdictNotTaken, outer_objective_band};
use ndarray::array;

/// The criterion's value at the seed: the scale of a REML criterion at
/// `n ≈ 10⁵`, where the relative floor is `1e-2`.
const V0_3018: f64 = 1.0e5;

/// `V(ρ) = V0 − ρ`: a constant gradient `−1`, a thousand times the claim band,
/// so no iterate is stationary and every filled window is a stall the guard
/// declared on cost alone.
fn value_3018(rho: f64) -> f64 {
    V0_3018 - rho
}

/// The claim band the guard's stationarity test applies; `rel_cost_tolerance =
/// 0` so the band does not widen with `|V|`.
const CLAIM_BAND_3018: f64 = 1.0e-3;

fn config_3018() -> OuterConfig {
    claim_band_config(CLAIM_BAND_3018)
}

/// The guard's relative floor, derived exactly as the ARC and trust-region
/// routes derive it.
fn rel_tol_3018(config: &OuterConfig) -> f64 {
    config
        .rel_cost_tolerance
        .unwrap_or(config.tolerance * 1.0e-2)
        .max(COST_STALL_REL_TOL_FLOOR)
}

/// The relative floor at the seed, `rel_tol·(1 + |V0|) = 1.00001e-2`.
fn rel_floor_3018() -> f64 {
    rel_tol_3018(&config_3018()) * (1.0 + V0_3018)
}

/// Resolvable progress the relative floor refuses: each step buys a quarter of
/// the floor, and an inner residual of `1e-9` puts the band sum at `2e-9`.
fn resolvable_3018() -> (f64, f64) {
    (0.25 * rel_floor_3018(), 1.0e-9)
}

/// Progress inside the band the relative floor admits: each step buys 2.5x the
/// floor, and an inner residual of twice the step puts the band sum above four
/// steps, so even a whole ARC window of them is not resolved.
fn unresolvable_3018() -> (f64, f64) {
    let step = 2.5 * rel_floor_3018();
    (step, 2.0 * step)
}

fn evidence_3018(inner_residual: f64) -> CertificateEvidence {
    CertificateEvidence {
        inner_residual: Some(InnerResidualCharge {
            energy: inner_residual,
            source: InnerResidualSource::InnerGradient,
        }),
        ..CertificateEvidence::default()
    }
}

/// Publish the evaluation's inner residual to the capture the bridge armed, as a
/// REML objective does. `None` publishes nothing, as a route with no residual
/// does.
fn publish_3018(inner_residual: Option<f64>) {
    if let Some(energy) = inner_residual {
        crate::estimate::outer_eval_capture::record_certificate_inner_residual(
            InnerResidualCharge {
                energy,
                source: InnerResidualSource::InnerGradient,
            },
        );
    }
}

/// `band_f` is the arithmetic's own rounding of `V` plus the inner mode's
/// residual error, from the evaluation's evidence, and is refused where that
/// evidence cannot form it.
#[test]
fn objective_band_is_formed_from_the_evaluations_own_evidence_3018() {
    let config = config_3018();
    let band = outer_objective_band(&config, V0_3018, &evidence_3018(1.0e-9))
        .expect("an evaluation that publishes its inner residual forms a band");
    assert_eq!(
        band.channels,
        gam_linalg::roundoff::accumulation_growth(1) * V0_3018,
        "a route that publishes no channels is charged one rounding of V"
    );
    assert_eq!(band.factor, 0.0);
    assert_eq!(band.inner_residual, 1.0e-9);
    assert_eq!(band.total(), band.channels + 1.0e-9);
    assert!(
        band.total() < 1.0e-8,
        "the band at |V| = 1e5 is {:.3e}, six orders under the relative floor {:.3e}",
        band.total(),
        rel_floor_3018(),
    );

    assert_eq!(
        outer_objective_band(&config, V0_3018, &CertificateEvidence::default()),
        Err(DecrementVerdictNotTaken::NoInnerResidual),
        "an evaluation with no inner residual carries an error the band cannot charge"
    );
    assert_eq!(
        outer_objective_band(&config, V0_3018, &evidence_3018(f64::NAN)),
        Err(DecrementVerdictNotTaken::NoInnerResidual),
    );

    let criterion = CertificateCriterion {
        cost: V0_3018,
        fixed_beta: V0_3018,
        logdet_h: 10.0,
        logdet_s: 0.0,
        kkt: 0.0,
        inner_residual_energy: Some(1.0e-9),
    };
    let with_criterion = CertificateEvidence {
        criterion: Some(criterion),
        ..evidence_3018(1.0e-9)
    };
    assert_eq!(
        outer_objective_band(&config, V0_3018, &with_criterion),
        Err(DecrementVerdictNotTaken::NoLogdetForwardError),
        "a log|H| channel whose factor derives no forward error is charged nothing"
    );
    let with_factor = CertificateEvidence {
        inner_factor: Some(InnerFactorCondition {
            logdet_forward_error: 4.0e-12,
        }),
        ..with_criterion.clone()
    };
    assert_eq!(
        outer_objective_band(&config, V0_3018, &with_factor),
        Err(DecrementVerdictNotTaken::NoProblemSize),
        "channels are charged at the formation count the problem size gives"
    );
    let sized = OuterConfig {
        problem_size: crate::rho_optimizer::OuterProblemSize {
            n_obs: Some(1_000),
            p_coefficients: Some(10),
        },
        ..config
    };
    let band = outer_objective_band(&sized, V0_3018, &with_factor)
        .expect("a sized route with a conditioned factor forms a band");
    assert_eq!(
        band.channels,
        gam_linalg::roundoff::accumulation_growth(1_000 + 100) * (V0_3018 + 10.0)
    );
    assert_eq!(band.factor, 2.0e-12);
    assert_eq!(band.inner_residual, 1.0e-9);
}

/// Drive the guard alone over `steps` accepted iterates of `V0 − k·step`, each
/// staged with the band its evidence forms (`None`: no evidence). Returns every
/// verdict.
fn drive_guard_3018(step: f64, inner_residual: Option<f64>, steps: usize) -> Vec<CostStallVerdict> {
    let config = config_3018();
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(rel_tol_3018(&config), COST_STALL_WINDOW, &config, exit);
    let band_at = |guard: &CostStallGuard, value: f64| {
        inner_residual.and_then(|energy| guard.objective_band(value, &evidence_3018(energy)))
    };
    let seed_band = band_at(&guard, V0_3018);
    guard.stage_objective_band(seed_band);
    guard.observe_seed(&array![0.0], V0_3018, 1.0);
    (1..=steps)
        .map(|k| {
            let rho = k as f64 * step;
            let value = value_3018(rho);
            let band = band_at(&guard, value);
            guard.stage_objective_band(band);
            guard.observe(&array![rho], value, 1.0, true)
        })
        .collect()
}

/// The guard's decision is the band sum where both values carry one, and the
/// relative floor where either does not, in both directions.
#[test]
fn the_guard_resolves_a_decrease_against_the_band_sum_3018() {
    let config = config_3018();
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let guard = CostStallGuard::new(rel_tol_3018(&config), COST_STALL_WINDOW, &config, exit);
    assert!(
        guard.stationarity_band() < 1.0,
        "the fixture's |g| = 1 must be outside the claim band, or every step is a stall"
    );
    let steps = 3 * COST_STALL_WINDOW;

    let (step, inner_residual) = resolvable_3018();
    let banded = drive_guard_3018(step, Some(inner_residual), steps);
    assert!(
        banded.iter().all(|verdict| matches!(verdict, CostStallVerdict::Continue)),
        "steps of {step:.3e} against a band sum of {:.3e} are resolved progress and never \
         fill a window: {banded:?}",
        2.0 * inner_residual,
    );
    let floored = drive_guard_3018(step, None, steps);
    assert!(
        matches!(
            floored[COST_STALL_WINDOW - 1],
            CostStallVerdict::StuckKeepDescending { .. }
        ),
        "the control: with no evidence the floor {:.3e} refuses every {step:.3e} step and \
         the window fills at step {COST_STALL_WINDOW}: {floored:?}",
        rel_floor_3018(),
    );

    let (step, inner_residual) = unresolvable_3018();
    let banded = drive_guard_3018(step, Some(inner_residual), steps);
    assert!(
        matches!(
            banded[COST_STALL_WINDOW - 1],
            CostStallVerdict::StuckKeepDescending { .. }
        ),
        "steps of {step:.3e} inside a band sum of {:.3e} are not resolved, and the window \
         fills at step {COST_STALL_WINDOW}: {banded:?}",
        4.0 * step,
    );
    let floored = drive_guard_3018(step, None, steps);
    assert!(
        floored.iter().all(|verdict| matches!(verdict, CostStallVerdict::Continue)),
        "the control: the floor {:.3e} admits every {step:.3e} step: {floored:?}",
        rel_floor_3018(),
    );
}

/// Drive the dense ARC bridge from the seed `ρ = 0` over `steps` accepted
/// iterates `ρ_k = k·step`, each evaluation publishing `inner_residual` to the
/// capture the bridge arms. The seed carries no band, as the run's seed does.
/// Returns every outcome up to the first stop and the guard's escape count.
fn drive_arc_3018(
    step: f64,
    inner_residual: Option<f64>,
    steps: usize,
) -> (Vec<Result<f64, String>>, usize) {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let config = config_3018();
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(value_3018(theta[0])),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        move |_: &mut (), theta: &Array1<f64>, order: OuterEvalOrder| {
            publish_3018(inner_residual);
            Ok(OuterEval {
                cost: value_3018(theta[0]),
                gradient: array![-1.0],
                hessian: match order {
                    OuterEvalOrder::ValueGradientHessian => HessianValue::Dense(array![[1.0]]),
                    _ => HessianValue::Unavailable,
                },
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(rel_tol_3018(&config), ARC_COST_STALL_WINDOW, &config, exit);
    guard.observe_second_order_seed(&array![0.0], V0_3018, 1.0, Some(true));
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    // No decrement verdict and no curvature stop: the guard's cost test alone
    // decides whether a window fills. The capture is armed for the guard.
    let mut bridge = OuterSecondOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        hessian_source: HessianSource::Analytic,
        eval_count: 0,
        outer_inner_cap: None,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some((array![-30.0], array![30.0])),
        curvature_stationary_floor: None,
        accepted_trials: AcceptedTrialGate::new(Arc::clone(&ledger)),
        decrement_verdict_config: None,
    };
    let mut outcomes = Vec::new();
    for iter in 0..steps {
        let rho = (iter + 1) as f64 * step;
        let outcome = eval_accepted_hessian_3017(&mut bridge, &ledger, &array![rho], iter)
            .map(|sample| sample.value)
            .map_err(|err| err.into_message());
        let stopped = outcome.is_err();
        outcomes.push(outcome);
        if stopped {
            break;
        }
    }
    let escapes = bridge
        .cost_stall
        .as_ref()
        .map_or(0, |guard| guard.stuck_escapes);
    (outcomes, escapes)
}

/// The dense ARC route: resolvable descent under the floor is no longer stopped
/// at `|g| = 1`, and descent inside the band no longer reads as progress.
#[test]
fn arc_stops_on_the_band_sum_not_the_relative_floor_3018() {
    let steps = 4 * ARC_COST_STALL_WINDOW;

    let (step, inner_residual) = resolvable_3018();
    let (banded, escapes) = drive_arc_3018(step, Some(inner_residual), steps);
    assert_eq!(banded.len(), steps, "resolved descent must not be stopped: {banded:?}");
    assert!(banded.iter().all(Result::is_ok), "{banded:?}");
    assert_eq!(
        escapes, 0,
        "steps of {step:.3e} its evaluations resolve to {:.3e} never fill a window",
        2.0 * inner_residual,
    );
    let (floored, _) = drive_arc_3018(step, None, steps);
    assert_eq!(
        floored.last(),
        Some(&Err(ARC_UNPROGRESSING_STALL_SENTINEL.to_string())),
        "the control: with no evidence the floor {:.3e} reads two windows of {step:.3e} \
         steps as bought nothing and stops the run at |g| = 1: {floored:?}",
        rel_floor_3018(),
    );
    assert_eq!(floored.len(), 2 * ARC_COST_STALL_WINDOW, "{floored:?}");

    let (step, inner_residual) = unresolvable_3018();
    let (banded, _) = drive_arc_3018(step, Some(inner_residual), steps);
    assert_eq!(
        banded.last(),
        Some(&Err(ARC_UNPROGRESSING_STALL_SENTINEL.to_string())),
        "windows of {step:.3e} steps inside a band sum of {:.3e} buy nothing the criterion \
         resolves, so the second one stops the run: {banded:?}",
        4.0 * step,
    );
    let (floored, escapes) = drive_arc_3018(step, None, steps);
    assert!(floored.iter().all(Result::is_ok), "the control never stops: {floored:?}");
    assert_eq!(escapes, 0, "the control: the floor admits every {step:.3e} step");
}

/// Drive the matrix-free trust-region bridge over the same trajectory as
/// [`drive_arc_3018`]. The route observes no seed, so the first iterate is the
/// incumbent and carries its own band.
fn drive_operator_3018(
    step: f64,
    inner_residual: Option<f64>,
    steps: usize,
) -> Vec<Result<f64, String>> {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either);
    let config = config_3018();
    let mut obj = problem.build_objective_with_eval_order(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(value_3018(theta[0])),
        |_: &mut (), _: &Array1<f64>| {
            Err(EstimationError::InvalidInput(
                "legacy eager eval should not run".to_string(),
            ))
        },
        move |_: &mut (), theta: &Array1<f64>, _: OuterEvalOrder| {
            publish_3018(inner_residual);
            Ok(OuterEval {
                cost: value_3018(theta[0]),
                gradient: array![-1.0],
                hessian: HessianValue::Dense(array![[1.0]]),
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let guard = CostStallGuard::new(
        rel_tol_3018(&config),
        ARC_COST_STALL_WINDOW,
        &config,
        Arc::new(Mutex::new(None)),
    );
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    let mut bridge = OuterOperatorBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        outer_inner_cap: None,
        eval_count: 0,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        cost_stall: Some(guard),
        cost_stall_bounds: Some((array![-30.0], array![30.0])),
        unprogressing_stop: Arc::new(Mutex::new(None)),
        accepted_trials: AcceptedTrialGate::new(Arc::clone(&ledger)),
    };
    let mut outcomes = Vec::new();
    for iter in 0..steps {
        let rho = (iter + 1) as f64 * step;
        let outcome = OperatorObjective::eval_value_grad_op(&mut bridge, &array![rho])
            .and_then(|sample| {
                report_accepted_trial_3017(&ledger, iter);
                bridge.settle_pending_trial().map_or(Ok(sample.value), Err)
            })
            .map_err(|err| err.into_message());
        let stopped = outcome.is_err();
        outcomes.push(outcome);
        if stopped {
            break;
        }
    }
    outcomes
}

/// The matrix-free trust-region route decides on the same band sum.
#[test]
fn the_operator_route_stops_on_the_band_sum_not_the_relative_floor_3018() {
    let steps = 4 * ARC_COST_STALL_WINDOW;

    let (step, inner_residual) = resolvable_3018();
    let banded = drive_operator_3018(step, Some(inner_residual), steps);
    assert_eq!(banded.len(), steps, "resolved descent must not be stopped: {banded:?}");
    assert!(banded.iter().all(Result::is_ok), "{banded:?}");
    let floored = drive_operator_3018(step, None, steps);
    assert!(
        floored.iter().any(Result::is_err),
        "the control: the floor {:.3e} stops {step:.3e} steps: {floored:?}",
        rel_floor_3018(),
    );

    let (step, inner_residual) = unresolvable_3018();
    let banded = drive_operator_3018(step, Some(inner_residual), steps);
    assert!(
        banded.iter().any(Result::is_err),
        "steps of {step:.3e} inside a band sum of {:.3e} buy nothing: {banded:?}",
        4.0 * step,
    );
    let floored = drive_operator_3018(step, None, steps);
    assert!(floored.iter().all(Result::is_ok), "the control never stops: {floored:?}");
}

/// Drive the first-order bridge from the seed `ρ = 0` over accepted iterates
/// `ρ_k = k·step`, each accepted by the ledger after its evaluation. Returns the
/// guard's escape count: the windows it filled at the non-stationary `|g| = 1`.
fn drive_first_order_3018(step: f64, inner_residual: Option<f64>, steps: usize) -> usize {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable);
    let config = config_3018();
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(value_3018(theta[0])),
        move |_: &mut (), theta: &Array1<f64>| {
            publish_3018(inner_residual);
            Ok(OuterEval {
                cost: value_3018(theta[0]),
                gradient: array![-1.0],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(rel_tol_3018(&config), COST_STALL_WINDOW, &config, exit);
    guard.observe_seed(&array![0.0], V0_3018, 1.0);
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    let mut bridge = OuterFirstOrderBridge {
        obj: &mut obj,
        layout: OuterThetaLayout::new(1, 0),
        outer_inner_cap: None,
        first_order_evals: 0,
        g_norm_initial: None,
        last_g_norm: None,
        last_value_grad_rho: None,
        value_probe_cache: Vec::new(),
        cost_stall: Some(guard),
        cost_stall_bounds: Some((array![-30.0], array![30.0])),
        consecutive_probe_refusals: 0,
        accepted_steps: Arc::clone(&ledger),
        pending_first_order: Vec::new(),
        incumbent: Some((array![0.0], V0_3018)),
        stratum_rank: None,
        stratum_probe: None,
    };
    // One evaluation past the last accept, so its fold is drained.
    for iter in 0..=steps {
        let rho = (iter + 1) as f64 * step;
        let evaluated = FirstOrderObjective::eval_grad(&mut bridge, &array![rho])
            .map(|_| ())
            .map_err(|err| err.into_message());
        assert_eq!(evaluated, Ok(()), "a non-stationary run must not halt");
        ledger.push(AcceptedOuterStep {
            iter,
            step_norm: step,
            actual_decrease: step,
        });
    }
    bridge
        .cost_stall
        .as_ref()
        .map_or(0, |guard| guard.stuck_escapes)
}

/// The first-order route decides on the same band sum.
#[test]
fn the_first_order_route_resolves_a_decrease_against_the_band_sum_3018() {
    let steps = 3 * COST_STALL_WINDOW;

    let (step, inner_residual) = resolvable_3018();
    assert_eq!(
        drive_first_order_3018(step, Some(inner_residual), steps),
        0,
        "steps of {step:.3e} its evaluations resolve never fill a window"
    );
    assert!(
        drive_first_order_3018(step, None, steps) > 0,
        "the control: the floor {:.3e} refuses every {step:.3e} step",
        rel_floor_3018(),
    );

    let (step, inner_residual) = unresolvable_3018();
    assert!(
        drive_first_order_3018(step, Some(inner_residual), steps) > 0,
        "steps of {step:.3e} inside a band sum of {:.3e} fill windows",
        4.0 * step,
    );
    assert_eq!(
        drive_first_order_3018(step, None, steps),
        0,
        "the control: the floor admits every {step:.3e} step"
    );
}
