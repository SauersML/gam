// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): the cost-stall guard judges a decrease against the objective bands
// of the two evaluations it compares, not a relative floor (#3018). Scope comes
// from the parent via `use super::*`.
//
// The defect these pin. The guard counted an accepted step as "no
// improvement" when it bought at most `rel_tol·(1 + |V|)`, with `rel_tol`
// floored at `1e-7`. Neither number is a property of the evaluations. At
// `|V| = 1e5` the floor was `1e-2`, so a run descending by `2.5e-3` per step,
// a decrease its evaluations resolve eleven orders of magnitude over, filled a
// window every `k` steps and the dense ARC route stopped it with
// `OUTER_ARC_UNPROGRESSING_STALL` at `|g| = 1`. The reverse held too: a
// decrease inside the inner mode's residual error cleared the floor and reset
// the window, so the guard read noise as progress.
//
// Where an evaluation publishes no band a value carries only its own rounding
// `γ₁|V|` (#3287), about `1.1e-11` at `V0 = 1e5`: the least error a computed
// value can have. The criterion's statistical resolution `τ_stat = 1/(2n)` is
// not that error (it is the certificate's decrement tolerance) and charging it
// stopped a search the arithmetic resolved as descending. The fixture declares
// `n = 50`, so `τ_stat = 1e-2`, the old floor, and it stays the step unit below:
// the no-evidence controls show a step far inside `τ_stat` is still resolved.
//
// Each computed value sits within its own `band_f` of the exact criterion, so
// a decrease is resolved exactly when it exceeds `band_f(V_k) + band_f(V_{k+1})`.
// An accepted step stalls when neither its measured nor
// its model's predicted decrease is resolved, and one stalled step reaches the
// verdict (#3018): there is no window.

use super::*;
use crate::estimate::outer_eval_capture::{
    CertificateCriterion, CertificateEvidence, InnerFactorCondition, InnerResidualCharge,
    InnerResidualSource,
};
use crate::rho_optimizer::decrement_bands::{DecrementVerdictNotTaken, outer_objective_band};
use ndarray::array;

/// The criterion's value at the seed: the scale of a REML criterion at
/// `n ≈ 10⁵`, where the old relative floor was `1e-2`.
const V0_3018: f64 = 1.0e5;

/// `V(ρ) = V0 − ρ`: a constant gradient `−1`, a thousand times the claim band,
/// so no iterate is stationary and every stall is one the guard declared on
/// cost alone.
fn value_3018(rho: f64) -> f64 {
    V0_3018 - rho
}

/// The claim band the guard's stationarity test applies.
const CLAIM_BAND_3018: f64 = 1.0e-3;

/// The declared observation count: `τ_stat = 1/(2·50) = 1e-2`.
const N_OBS_3018: usize = 50;

fn config_3018() -> OuterConfig {
    OuterConfig {
        problem_size: crate::rho_optimizer::OuterProblemSize {
            n_obs: Some(N_OBS_3018),
            p_coefficients: Some(1),
            information_count: None,
        },
        ..claim_band_config(CLAIM_BAND_3018)
    }
}

/// The resolution a value carries when its evaluation publishes no evidence:
/// its own rounding `γ₁|V|` (#3287), which moves with `|V|`.
fn fallback_3018(value: f64) -> f64 {
    value_representation_band(value)
}

/// The criterion's statistical resolution `τ_stat = 1/(2n) = 1e-2`: the step
/// unit of the trajectories below, and the old floor's value at `|V| = 1e5`.
fn tau_3018() -> f64 {
    crate::rho_optimizer::outer_criterion_resolution(&config_3018())
}

/// Resolvable progress the old floor refused: each step buys a quarter of `τ`,
/// and an inner residual of `1e-9` puts the band sum at `2e-9`. Without
/// evidence the pair of roundings is `2γ₁|V0| ≈ 2.2e-11`, so the step is
/// resolved there too.
fn resolvable_3018() -> (f64, f64) {
    (0.25 * tau_3018(), 1.0e-9)
}

/// One unit in the last place of `V0`: a decrease of `≈ 1.46e-11`, inside the
/// pair of roundings `2γ₁|V0| ≈ 2.2e-11` two unbanded values carry.
fn sub_rounding_step_3018() -> f64 {
    let step = V0_3018 - f64::from_bits(V0_3018.to_bits() - 1);
    assert!(
        step < 2.0 * fallback_3018(V0_3018),
        "a last-place step {step:.3e} must sit inside the pair of roundings {:.3e}",
        2.0 * fallback_3018(V0_3018),
    );
    step
}

/// Progress inside the band an inner residual forms: each step buys 2.5x `τ`,
/// and an inner residual of twice the step puts the band sum above four
/// steps. Without evidence the step clears the pair of roundings.
fn unresolvable_3018() -> (f64, f64) {
    let step = 2.5 * tau_3018();
    (step, 2.0 * step)
}

/// The resolution of a value `V` whose evaluation published `inner_residual`
/// (`None`: nothing), as every bridge charges it.
fn resolution_3018(value: f64, inner_residual: Option<f64>) -> f64 {
    let evidence = inner_residual.map_or_else(CertificateEvidence::default, evidence_3018);
    sample_resolution(&config_3018(), value, &evidence)
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
    let config = claim_band_config(CLAIM_BAND_3018);
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
        "the band at |V| = 1e5 is {:.3e}, six orders under the statistical resolution {:.3e}",
        band.total(),
        tau_3018(),
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
            information_count: None,
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
/// carrying the resolution its evidence gives (`None`: no evidence, so its own
/// rounding `γ₁|V|`).
/// Each step's linear model predicts exactly its decrease, `−gᵀs = step`, as a
/// line search on `V0 − ρ` does. Returns every verdict.
fn drive_guard_3018(step: f64, inner_residual: Option<f64>, steps: usize) -> Vec<CostStallVerdict> {
    let config = config_3018();
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(&config, exit);
    guard.observe_seed(&array![0.0], V0_3018, resolution_3018(V0_3018, inner_residual), 1.0);
    (1..=steps)
        .map(|k| {
            let rho = array![k as f64 * step];
            let value = value_3018(rho[0]);
            guard.observe(
                StallSample {
                    point: &rho,
                    value,
                    resolution: resolution_3018(value, inner_residual),
                    grad_norm: 1.0,
                    trusted: true,
                    curvature_psd: None,
                },
                step,
            )
        })
        .collect()
}

/// The guard's decision is the pair of resolutions the two values carry: their
/// bands where the evaluations publish evidence, their roundings where they do
/// not, in both directions. A step that is not resolved, and whose model
/// promised no more, stalls at once.
#[test]
fn the_guard_resolves_a_decrease_against_the_band_sum_3018() {
    let config = config_3018();
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let guard = CostStallGuard::new(&config, exit);
    assert!(
        guard.stationarity_band() < 1.0,
        "the fixture's |g| = 1 must be outside the claim band, or every step is a stall"
    );
    let steps = 18;

    let (step, inner_residual) = resolvable_3018();
    let banded = drive_guard_3018(step, Some(inner_residual), steps);
    assert!(
        banded.iter().all(|verdict| matches!(verdict, CostStallVerdict::Continue)),
        "steps of {step:.3e} against a band sum of {:.3e} are resolved progress and never \
         stall: {banded:?}",
        2.0 * inner_residual,
    );
    let floored = drive_guard_3018(step, None, steps);
    assert!(
        floored.iter().all(|verdict| matches!(verdict, CostStallVerdict::Continue)),
        "the control: with no evidence each value carries only its rounding {:.3e}, so \
         {step:.3e} steps, a quarter of τ = {:.3e}, are resolved and never stall (#3287): \
         {floored:?}",
        fallback_3018(V0_3018),
        tau_3018(),
    );
    let step = sub_rounding_step_3018();
    let floored = drive_guard_3018(step, None, steps);
    assert!(
        matches!(floored[0], CostStallVerdict::StuckKeepDescending { .. }),
        "with no evidence a last-place step {step:.3e}, predicted as {step:.3e}, sits inside \
         the pair of roundings {:.3e} and the first one stalls: {floored:?}",
        2.0 * fallback_3018(V0_3018),
    );

    let (step, inner_residual) = unresolvable_3018();
    let banded = drive_guard_3018(step, Some(inner_residual), steps);
    assert!(
        matches!(banded[0], CostStallVerdict::StuckKeepDescending { .. }),
        "steps of {step:.3e} inside a band sum of {:.3e} are not resolved, and the first \
         one stalls: {banded:?}",
        4.0 * step,
    );
    let floored = drive_guard_3018(step, None, steps);
    assert!(
        floored.iter().all(|verdict| matches!(verdict, CostStallVerdict::Continue)),
        "the control: the rounding {:.3e} admits every {step:.3e} step: {floored:?}",
        fallback_3018(V0_3018),
    );
}

/// Drive the dense ARC bridge from the seed `ρ = 0` over `steps` accepted
/// iterates `ρ_k = k·step`, each evaluation publishing `inner_residual` to the
/// capture the bridge arms. The seed carries the resolution its own evidence
/// gives, as the run's seed does. Each step is reported accepted by a model that
/// predicted no decrease, so its measured decrease alone decides it. Returns
/// every outcome up to the first stop and the guard's escape count.
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
    let mut guard = CostStallGuard::new(&config, exit);
    guard.observe_second_order_seed(
        &array![0.0],
        V0_3018,
        resolution_3018(V0_3018, inner_residual),
        1.0,
        Some(true),
    );
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    // No decrement verdict and no curvature stop: the guard's cost test alone
    // decides whether a step stalls. The capture is armed for the guard.
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
        curvature_stationary_resolution: None,
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
        .map_or(0, CostStallGuard::stuck_escapes);
    (outcomes, escapes)
}

/// The dense ARC route: resolvable descent is no longer stopped at `|g| = 1`,
/// with or without evidence, and descent inside the band no longer reads as
/// progress.
#[test]
fn arc_stops_on_the_band_sum_not_the_relative_floor_3018() {
    let steps = 12;

    let (step, inner_residual) = resolvable_3018();
    let (banded, escapes) = drive_arc_3018(step, Some(inner_residual), steps);
    assert_eq!(banded.len(), steps, "resolved descent must not be stopped: {banded:?}");
    assert!(banded.iter().all(Result::is_ok), "{banded:?}");
    assert_eq!(
        escapes, 0,
        "steps of {step:.3e} its evaluations resolve to {:.3e} never stall",
        2.0 * inner_residual,
    );
    // With no evidence each value carries only its rounding `γ₁|V|`, so a step of
    // a quarter of `τ` is resolved nine orders of magnitude over the pair and
    // never stalls. Charging `τ` instead stopped this run at the second step, at
    // |g| = 1 (#3287).
    let (floored, escapes) = drive_arc_3018(step, None, steps);
    assert_eq!(floored.len(), steps, "the control must not be stopped: {floored:?}");
    assert!(floored.iter().all(Result::is_ok), "{floored:?}");
    assert_eq!(
        escapes, 0,
        "the control: {step:.3e} steps against the pair of roundings {:.3e} never stall",
        2.0 * fallback_3018(V0_3018),
    );

    let (step, inner_residual) = unresolvable_3018();
    let (banded, _) = drive_arc_3018(step, Some(inner_residual), steps);
    assert_eq!(
        banded.last(),
        Some(&Err(ARC_UNPROGRESSING_STALL_SENTINEL.to_string())),
        "{step:.3e} steps inside a band sum of {:.3e} buy nothing the criterion resolves, \
         so the second stall stops the run: {banded:?}",
        4.0 * step,
    );
    let (floored, escapes) = drive_arc_3018(step, None, steps);
    assert!(floored.iter().all(Result::is_ok), "the control never stops: {floored:?}");
    assert_eq!(escapes, 0, "the control: the rounding admits every {step:.3e} step");
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
    let guard = CostStallGuard::new(&config, Arc::new(Mutex::new(None)));
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
    let steps = 12;

    let (step, inner_residual) = resolvable_3018();
    let banded = drive_operator_3018(step, Some(inner_residual), steps);
    assert_eq!(banded.len(), steps, "resolved descent must not be stopped: {banded:?}");
    assert!(banded.iter().all(Result::is_ok), "{banded:?}");
    let floored = drive_operator_3018(step, None, steps);
    assert_eq!(floored.len(), steps, "the control must not be stopped: {floored:?}");
    assert!(
        floored.iter().all(Result::is_ok),
        "the control: the rounding {:.3e} resolves every {step:.3e} step (#3287): {floored:?}",
        fallback_3018(V0_3018),
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
/// `ρ_k = k·step`, each accepted by the ledger after its evaluation with the
/// decrease its linear model predicted, `−gᵀs = step`. The seed carries the
/// resolution its own evidence gives. Returns the guard's escape count (the
/// stalls it let continue at the non-stationary `|g| = 1`) and whether the run
/// stopped.
fn drive_first_order_3018(step: f64, inner_residual: Option<f64>, steps: usize) -> (usize, bool) {
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
    let mut guard = CostStallGuard::new(&config, exit);
    guard.observe_seed(&array![0.0], V0_3018, resolution_3018(V0_3018, inner_residual), 1.0);
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
        accepted_steps: Arc::clone(&ledger),
        pending_first_order: Vec::new(),
        incumbent: Some(OuterIncumbent {
            rho: array![0.0],
            cost: V0_3018,
            gradient: array![-1.0],
        }),
        stratum_rank: None,
        stratum_probe: None,
    };
    // One evaluation past the last accept, so its fold is drained.
    let mut stopped = false;
    for iter in 0..=steps {
        let rho = (iter + 1) as f64 * step;
        if FirstOrderObjective::eval_grad(&mut bridge, &array![rho]).is_err() {
            stopped = true;
            break;
        }
        ledger.push(AcceptedOuterStep {
            iter,
            step_norm: step,
            actual_decrease: step,
            predicted_decrease: step,
        });
    }
    let escapes = bridge
        .cost_stall
        .as_ref()
        .map_or(0, CostStallGuard::stuck_escapes);
    (escapes, stopped)
}

/// The first-order route decides on the same pair of resolutions. A run whose
/// every step stalls is granted an escape at each stall at `|g| = 1`, and the
/// licence stops it at the second, since it bought no resolved descent in
/// between (#2817). With no evidence a value carries its rounding, so only a
/// step inside that rounding stalls (#3287).
#[test]
fn the_first_order_route_resolves_a_decrease_against_the_band_sum_3018() {
    let steps = 18;

    let (step, inner_residual) = resolvable_3018();
    assert_eq!(
        drive_first_order_3018(step, Some(inner_residual), steps),
        (0, false),
        "steps of {step:.3e} its evaluations resolve never stall"
    );
    assert_eq!(
        drive_first_order_3018(step, None, steps),
        (0, false),
        "the control: each value carries only its rounding {:.3e}, so every {step:.3e} \
         step is resolved (#3287)",
        fallback_3018(V0_3018),
    );
    let step = sub_rounding_step_3018();
    assert_eq!(
        drive_first_order_3018(step, None, steps),
        (2, true),
        "a last-place step {step:.3e}, predicted as {step:.3e}, inside the pair of roundings \
         {:.3e} stalls",
        2.0 * fallback_3018(V0_3018),
    );

    let (step, inner_residual) = unresolvable_3018();
    assert_eq!(
        drive_first_order_3018(step, Some(inner_residual), steps),
        (2, true),
        "steps of {step:.3e} inside a band sum of {:.3e} stall",
        4.0 * step,
    );
    assert_eq!(
        drive_first_order_3018(step, None, steps),
        (0, false),
        "the control: the pair of roundings admits every {step:.3e} step"
    );
}

/// Drive the guard over `steps` accepted iterates of `V0 + shift − k·step`, each
/// value charged the fixed resolution `τ` and each step predicted by its linear
/// model as `step`, returning each verdict's kind.
fn drive_shifted_guard_3018(
    shift: f64,
    step: f64,
    steps: usize,
) -> Vec<std::mem::Discriminant<CostStallVerdict>> {
    let config = config_3018();
    let tau = tau_3018();
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(&config, exit);
    guard.observe_seed(&array![0.0], V0_3018 + shift, tau, 1.0);
    (1..=steps)
        .map(|k| {
            let rho = array![k as f64 * step];
            std::mem::discriminant(&guard.observe(
                StallSample {
                    point: &rho,
                    value: value_3018(rho[0]) + shift,
                    resolution: tau,
                    grad_norm: 1.0,
                    trusted: true,
                    curvature_psd: None,
                },
                step,
            ))
        })
        .collect()
}

/// The criterion's additive constant carries no information about which `ρ` is
/// better, so a decrease is resolved or not whatever `|V|` is: shifting every
/// value by `C` leaves every verdict unchanged. Under the old resolution
/// `rel·(1 + |V|)` a shift of `10⁷` moved the floor from `10⁻²` to `1`, so a
/// step of `4τ` that the unshifted run resolved was refused once shifted.
#[test]
fn the_guard_verdicts_are_invariant_to_an_additive_shift_of_the_criterion_3018() {
    let steps = 18;
    let tau = tau_3018();
    for &step in &[4.0 * tau, 0.25 * tau] {
        let reference = drive_shifted_guard_3018(0.0, step, steps);
        for &shift in &[-9.9e4, 1.0e7, 1.0e9] {
            assert_eq!(
                drive_shifted_guard_3018(shift, step, steps),
                reference,
                "step {step:.3e}: shifting the criterion by {shift:.1e} must not change a verdict"
            );
        }
    }
    let resolved = drive_shifted_guard_3018(1.0e9, 4.0 * tau, steps);
    assert!(
        resolved
            .iter()
            .all(|kind| *kind == std::mem::discriminant(&CostStallVerdict::Continue)),
        "steps of 4τ are resolved at |V| = 1e9 and never stall"
    );
    // A step of τ/4, predicted as τ/4, is resolved neither measured nor predicted
    // against the pair `2τ`, so the first one stalls (#3018), at |V| = 1e9 as at
    // |V| = 1e5.
    let unresolved = drive_shifted_guard_3018(1.0e9, 0.25 * tau, steps);
    assert_ne!(
        unresolved[0],
        std::mem::discriminant(&CostStallVerdict::Continue),
        "steps of τ/4 are refused at |V| = 1e9 as at |V| = 1e5"
    );
}

/// A STEP'S DECREASE IS NOT THE DECREASE LEFT (#3286). On `V = V0 − ρ` the search
/// buys `¼·τ` per step with its descent unbounded. Every step is below the
/// resolution the certificate's verdict and the online stops judge the decrease
/// LEFT at ([`outer_resolution`](crate::rho_optimizer::decrement_bands::outer_resolution)),
/// and not one of them is a stall: the guard judges each step's decrease against
/// the two values' own bands. Judging a step by the decrease-left resolution
/// would halt this search at `|g| = 1`.
#[test]
fn a_step_below_the_decrease_left_resolution_is_still_progress_3286() {
    let (step, inner_residual) = resolvable_3018();
    let pair = 2.0 * resolution_3018(V0_3018, Some(inner_residual));
    let decrease_left = crate::rho_optimizer::decrement_bands::outer_resolution(tau_3018(), pair);
    assert!(
        step < decrease_left,
        "fixture premise: each {step:.3e} step buys less than the decrease-left resolution \
         {decrease_left:.3e}"
    );
    let verdicts = drive_guard_3018(step, Some(inner_residual), 18);
    assert!(
        verdicts
            .iter()
            .all(|verdict| matches!(verdict, CostStallVerdict::Continue)),
        "steps the bands resolve are progress whatever the decrease-left resolution is: \
         {verdicts:?}"
    );
}

/// An unbanded value on a route that declares no size carries its own rounding, not
/// zero (#3286). With no observation count `τ = 0`, so a value whose evaluation forms
/// no band used to carry a resolution of `0`: every difference then read as resolved
/// progress, and a flat criterion never stalled. It now carries `γ₁·|V|`, so a step
/// buying less than the pair of those, with a model that promised nothing, stalls.
#[test]
fn an_unbanded_value_with_no_declared_size_carries_its_own_rounding_3286() {
    let config = claim_band_config(CLAIM_BAND_3018);
    let tau = crate::rho_optimizer::outer_criterion_resolution(&config);
    assert_eq!(tau, 0.0, "fixture premise: the route declares no size");
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(&config, exit);
    let own = guard.value_resolution(V0_3018, &CertificateEvidence::default());
    assert_eq!(
        own,
        crate::rho_optimizer::decrement_bands::value_representation_band(V0_3018),
        "an unbanded value's resolution is its own representation error"
    );
    guard.observe_seed(&array![0.0], V0_3018, own, 1.0);
    let rho = array![1.0e-3];
    // One unit in the last place below `V0`: a decrease the arithmetic represents,
    // `1.46e-11`, inside the pair of representation errors `2.2e-11`.
    let value = V0_3018.next_down();
    let verdict = guard.observe(
        StallSample {
            point: &rho,
            value,
            resolution: guard.value_resolution(value, &CertificateEvidence::default()),
            grad_norm: 1.0,
            trusted: true,
            curvature_psd: None,
        },
        0.0,
    );
    assert!(
        !matches!(verdict, CostStallVerdict::Continue),
        "a decrease of {:.3e} inside the two values' rounding is not progress: {verdict:?}",
        V0_3018 - value,
    );
}
