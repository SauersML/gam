// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): the outer stationarity band — the #2613 gradient-only stiff-ridge
// replay, the cost-stall window counting accepted steps, the band's anchor,
// and the #2458 derived standard on routes with no declared curvature. Scope
// comes from the parent via `use super::*`; the split is purely physical.

use super::*;
use ndarray::array;

/// The certificate band alone. #2688 moved the rung into the production return
/// type on purpose, so the value-only form lives in the tests that compare
/// bands to each other rather than beside the thing it was extracted from.
fn outer_stationarity_band_at(config: &OuterConfig, cost_at_point: f64) -> f64 {
    outer_stationarity_band_and_rung_at(config, cost_at_point).bound
}

// ─── #2613 diagnostic (zz_measure): the gradient-only stiff-ridge trajectory ──

/// Replay the #2392 recovery objective with every outer evaluation logged, so
/// the line-search path between the pull-back seed and the 12.0258 stall is
/// visible instead of inferred. Signal test: it makes no claim about the
/// optimum, only that the audit refuses the wrong rail, publishes an inward
/// pull-back, and that the recovery run terminates; the `eprintln` stream is
/// the output.
///
/// It RUNS. `#[ignore]` is a build-stopper here (`build.rs`'s ban scanner), and
/// the reason that rule exists is exactly this shape: an ignored test is one
/// nobody notices going red. The assertions above the print stream — the
/// `expect_err` on the upper rail and the `expect` on the reseed — are the
/// contract, and they hold at the tree this landed on.
#[test]
fn zz_measure_2613_gradient_only_stiff_ridge_trajectory() {
    const AMPLITUDE: f64 = 1.0e4;
    const RHO_STAR: f64 = 12.0;

    // A second `try_init` in one process is an `Err`, and that is the expected
    // state whenever another test installed the logger first. Either way trace
    // output is reachable, which is all this diagnostic needs; the result is
    // reported rather than discarded.
    if env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .is_test(false)
        .try_init()
        .is_err()
    {
        log::debug!("zz_measure #2613: a logger was already installed by another test");
    }

    let calls = Arc::new(Mutex::new(Vec::<(char, f64, f64, f64)>::new()));
    let cost_log = Arc::clone(&calls);
    let eval_log = Arc::clone(&calls);

    let cost = move |rho: &Array1<f64>| {
        let q = (RHO_STAR - rho[0]).exp();
        let v = AMPLITUDE * (-q + 0.5 * q * q);
        cost_log
            .lock()
            .expect("log")
            .push(('c', rho[0], v, f64::NAN));
        v
    };
    let eval = move |rho: &Array1<f64>| {
        let q = (RHO_STAR - rho[0]).exp();
        let v = AMPLITUDE * (-q + 0.5 * q * q);
        let g = AMPLITUDE * (q - q * q);
        eval_log.lock().expect("log").push(('g', rho[0], v, g));
        OuterEval {
            cost: v,
            gradient: array![g],
            hessian: HessianValue::Unavailable,
            inner_beta_hint: Some(array![q]),
        }
    };

    let audit_problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable);
    let audit_cost = cost.clone();
    let audit_eval = eval.clone();
    let mut audit_obj = audit_problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(audit_cost(rho)),
        move |_: &mut (), rho: &Array1<f64>| Ok(audit_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let refusal = audit_stationary_point(
        &mut audit_obj,
        array![29.9],
        "gradient-only wrong-rail audit #2613",
    )
    .expect_err("the inward-descent upper rail must not certify");
    let reseed = refusal
        .result
        .wrong_rail_reseed
        .expect("first-order clean-tail evidence must publish an inward pull-back");
    eprintln!("[zz_measure #2613] pull-back reseed = {reseed:?}");
    calls.lock().expect("log").clear();

    let recovery_problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(reseed)
        .with_screen_initial_rho(false)
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        });
    let mut recovery_obj = recovery_problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(cost(rho)),
        move |_: &mut (), rho: &Array1<f64>| Ok(eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let outcome = recovery_problem.run(&mut recovery_obj, "zz_measure #2613 recovery");
    for (idx, (kind, rho, value, grad)) in calls.lock().expect("log").iter().enumerate() {
        eprintln!(
            "[zz_measure #2613] {idx:>4} {kind} rho={rho:+.12e} V={value:+.12e} g={grad:+.6e}"
        );
    }
    match outcome {
        Ok(result) => eprintln!(
            "[zz_measure #2613] OK rho={:?} iters={}",
            result.rho, result.iterations
        ),
        Err(err) => eprintln!("[zz_measure #2613] ERR {err}"),
    }
}

// ─── #2613 the cost-stall guard counts ACCEPTED steps, not evaluations ────────

/// `‖Pg‖` at every point in the #2613 window tests: below the guard's `1e-3`
/// stationarity threshold, so a filled window certifies as `Converged` rather
/// than routing through the `StuckKeepDescending` escape budget. The escape
/// ladder is a different mechanism with its own tests; keeping it out of these
/// makes the halt index a clean function of the window alone.
const STATIONARY_GRAD_2613: f64 = 5.0e-4;

/// The plateau every #2613 window test sits on: a Strong-Wolfe zoom's trials
/// converging geometrically to one point, so consecutive costs differ by ~1e-9
/// against a `1e-7 · (1 + 4996.7) ≈ 5e-4` improvement floor while the ITERATE
/// has not moved once.
fn zoom_plateau_schedule_2613(len: usize) -> Vec<(f64, f64, f64)> {
    (0..len)
        .map(|i| {
            let shrink = 0.5_f64.powi(i as i32);
            (
                -4996.7 + 1.0e-9 * shrink,
                12.02577 + 1.0e-6 * shrink,
                STATIONARY_GRAD_2613,
            )
        })
        .collect()
}

/// Drive a first-order bridge over a `(cost, ρ, ‖g‖)` schedule, one entry per
/// `eval_grad`, against a guard seeded at `(seed_rho, seed_cost, seed_grad)`.
///
/// `ledger` is the accepted-step channel `OuterAcceptObserver` writes to in
/// production; a test drives it through `accept_after` so an "accepted step" is
/// exactly what `opt` would have reported and nothing more. Returns each
/// evaluation's outcome (stopping at the first error) and whatever the guard
/// published into its shared exit cell.
fn drive_first_order_bridge_2613(
    schedule: Vec<(f64, f64, f64)>,
    seed: (Array1<f64>, f64, f64),
    ledger: Option<Arc<AcceptedStepLedger>>,
    mut accept_after: impl FnMut(usize, f64),
) -> (Vec<Result<f64, String>>, Option<CostStallExit>) {
    let (seed_rho, seed_cost, seed_grad) = seed;
    let calls = Arc::new(AtomicUsize::new(0));
    let table = Arc::new(schedule.clone());
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(0.0),
        {
            let calls = Arc::clone(&calls);
            let table = Arc::clone(&table);
            move |_: &mut (), _: &Array1<f64>| {
                let idx = calls.fetch_add(1, Ordering::Relaxed);
                let (cost, _, grad) = table[idx.min(table.len() - 1)];
                Ok(OuterEval {
                    cost,
                    gradient: array![grad],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let exit: Arc<Mutex<Option<CostStallExit>>> = Arc::new(Mutex::new(None));
    let mut guard = CostStallGuard::new(1.0e-7, COST_STALL_WINDOW, 1.0e-3, exit.clone());
    guard.observe_seed(&seed_rho, seed_cost, seed_grad);
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
        accepted_steps: ledger,
        pending_first_order: Vec::new(),
        incumbent: Some((seed_rho, seed_cost)),
    };
    let mut outcomes = Vec::new();
    for (idx, (cost, rho, _)) in schedule.iter().enumerate() {
        match FirstOrderObjective::eval_grad(&mut bridge, &array![*rho]) {
            Ok(sample) => outcomes.push(Ok(sample.value)),
            Err(err) => {
                outcomes.push(Err(err.into_message()));
                break;
            }
        }
        accept_after(idx, *cost);
    }
    let published = exit.lock().expect("exit cell").take();
    (outcomes, published)
}

/// #2613 — a Strong-Wolfe zoom bisecting toward a point emits a run of gradient
/// evaluations whose costs differ by less than the stall floor. None of them is
/// an accepted outer step, so none may advance the cost-stall window.
///
/// This is the defect from the opposite side of the one
/// `wrong_rail_pullback_recovers_gradient_only_objective_2392` exercises:
/// there, the spurious halt stopped a solve that was about to succeed; here the
/// bridge is driven directly with the evaluation pattern a zoom emits and must
/// stay silent no matter how long the zoom runs. Pairs with
/// [`accepted_steps_still_trip_the_cost_stall_window_2613`], which feeds the
/// IDENTICAL schedule with accept signals attached — the two differ in nothing
/// else, which is the whole content of the fix.
#[test]
fn line_search_probes_never_advance_the_cost_stall_window_2613() {
    let schedule = zoom_plateau_schedule_2613(COST_STALL_WINDOW * 4);
    let offered: Arc<Mutex<Vec<(usize, f64)>>> = Arc::new(Mutex::new(Vec::new()));
    let (outcomes, published) = drive_first_order_bridge_2613(
        schedule.clone(),
        (array![12.02577], -4996.7, STATIONARY_GRAD_2613),
        Some(Arc::default()),
        {
            // No accepted steps: `opt` is still inside iteration 0. Record what
            // the drive offers instead of discarding it, so "every probe
            // reached the accept hook and none of them became an accept" is
            // CHECKED below rather than asserted by an empty body.
            let offered = Arc::clone(&offered);
            move |idx, cost| {
                offered.lock().expect("offered ledger").push((idx, cost));
            }
        },
    );
    assert_eq!(
        offered.lock().expect("offered ledger").len(),
        outcomes.iter().filter(|outcome| outcome.is_ok()).count(),
        "the accept hook must be offered exactly the successful evaluations"
    );
    assert_eq!(
        outcomes.len(),
        schedule.len(),
        "the guard halted a line search mid-zoom after {} of {} probes",
        outcomes.len(),
        schedule.len(),
    );
    assert!(
        outcomes.iter().all(Result::is_ok),
        "no line-search probe may produce the cost-stall sentinel: {outcomes:?}",
    );
    // `observe_seed` publishes the seed up front so the budget-exhaustion path
    // always has a feasible fallback (#1371), so the cell is never empty. What
    // must not happen is the probes DISPLACING that seed or counting as steps.
    let published = published.expect("the seed is published up front");
    assert_eq!(
        published.iterations, 1,
        "{} probes advanced the accepted-iterate count: {published:?}",
        schedule.len(),
    );
    assert_eq!(
        published.rho,
        array![12.02577],
        "a probe displaced the seed incumbent: {published:?}",
    );

    // And the same schedule down the pre-#2613 path — `accepted_steps: None`,
    // i.e. fold every gradient evaluation — DOES halt, which is what this test
    // is defending against. Without this the assertions above would pass on a
    // guard that had simply been disabled.
    let legacy_offered: Arc<Mutex<Vec<(usize, f64)>>> = Arc::new(Mutex::new(Vec::new()));
    let (legacy, _) = drive_first_order_bridge_2613(
        schedule.clone(),
        (array![12.02577], -4996.7, STATIONARY_GRAD_2613),
        None,
        {
            // Same "no accepted steps" signal as the sibling above, and checked
            // the same way rather than written as an empty body.
            let legacy_offered = Arc::clone(&legacy_offered);
            move |idx, cost| {
                legacy_offered
                    .lock()
                    .expect("legacy offered ledger")
                    .push((idx, cost));
            }
        },
    );
    assert_eq!(
        legacy_offered.lock().expect("legacy offered ledger").len(),
        legacy.iter().filter(|outcome| outcome.is_ok()).count(),
        "the legacy accept hook must be offered exactly the successful evaluations"
    );
    // `COST_STALL_WINDOW − 1`, not `COST_STALL_WINDOW`: the inline fold has no
    // accept latency, so the sixth observation lands on the sixth evaluation
    // rather than the seventh.
    assert_eq!(
        legacy.iter().position(Result::is_err),
        Some(COST_STALL_WINDOW - 1),
        "folding every gradient eval must reach the sentinel — else this test proves nothing \
         about the accept gating: {legacy:?}",
    );
}

/// #2613 — the guard's own job is untouched: a genuine run of accepted outer
/// steps with no improvement still halts, on exactly the
/// `COST_STALL_WINDOW`-th accepted step, and still certifies a stationary
/// plateau as converged.
#[test]
fn accepted_steps_still_trip_the_cost_stall_window_2613() {
    let schedule = zoom_plateau_schedule_2613(COST_STALL_WINDOW * 4);
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    let incumbent = Arc::new(Mutex::new(-4996.7_f64));
    let (outcomes, published) = {
        let ledger = Arc::clone(&ledger);
        let incumbent = Arc::clone(&incumbent);
        drive_first_order_bridge_2613(
            schedule.clone(),
            (array![12.02577], -4996.7, STATIONARY_GRAD_2613),
            Some(Arc::clone(&ledger)),
            move |idx, cost| {
                let mut prev = incumbent.lock().expect("incumbent");
                ledger.push(AcceptedOuterStep {
                    iter: idx,
                    step_norm: 1.0e-6,
                    actual_decrease: *prev - cost,
                });
                *prev = cost;
            },
        )
    };
    let halted = outcomes
        .iter()
        .position(Result::is_err)
        .expect("a stalled run of accepted steps must halt");
    assert_eq!(
        outcomes[halted].as_ref().unwrap_err(),
        COST_STALL_CONVERGED_SENTINEL,
        "the halt must use the shared cost-stall sentinel",
    );
    // The accept for evaluation `i` is published after it and drained at the
    // top of evaluation `i+1`, so the window closes on evaluation
    // `COST_STALL_WINDOW`. One evaluation of latency is inherent:
    // `on_step_accepted` fires after the line search that produced the step.
    assert_eq!(
        halted, COST_STALL_WINDOW,
        "the window must close on the {COST_STALL_WINDOW}th accepted step: {outcomes:?}",
    );
    let published = published.expect("a stalled run must publish its best iterate");
    assert!(
        published.converged,
        "a plateau whose |Pg| = {STATIONARY_GRAD_2613:.1e} clears the 1e-3 band is a stationary \
         optimum, not a floor: {published:?}",
    );
}

/// #2613 — `opt::StepInfo` carries no point, so the bridge reconstructs the
/// accepted cost from `actual_decrease` and matches it against the evaluations
/// it made. The match must be by COST, not "the most recent evaluation":
/// `opt`'s coordinate rescue evaluates further points AFTER the line search
/// returns and before `on_step_accepted` fires, and folding one of those would
/// put a rejected probe into the guard's window under the accepted step's name.
#[test]
fn accepted_step_resolves_by_cost_not_by_recency_2613() {
    // Evaluation 0 is the accepted trial; 1 and 2 are rescue pokes that lose.
    // Then a plateau at the accepted cost, long enough to close the window.
    let mut schedule = vec![
        (-100.0, 11.0, STATIONARY_GRAD_2613),
        (-99.0, 11.5, STATIONARY_GRAD_2613),
        (-98.5, 10.5, STATIONARY_GRAD_2613),
    ];
    schedule.extend(
        (0..(COST_STALL_WINDOW + 3)).map(|_| (-100.0, 11.0, STATIONARY_GRAD_2613)),
    );
    let ledger: Arc<AcceptedStepLedger> = Arc::default();
    let (outcomes, published) = {
        let ledger = Arc::clone(&ledger);
        drive_first_order_bridge_2613(
            schedule.clone(),
            (array![16.9], -10.0, STATIONARY_GRAD_2613),
            Some(Arc::clone(&ledger)),
            move |idx, cost| {
                if idx < 2 {
                    // Inside one line search plus its rescue: nothing accepted yet.
                    return;
                }
                ledger.push(AcceptedOuterStep {
                    iter: idx,
                    step_norm: 5.9,
                    // `f_k − f_next` for the point evaluated FIRST. On the first
                    // accept the incumbent is the seed; afterwards the plateau
                    // repeats the accepted cost, so every later step decreases
                    // by nothing.
                    actual_decrease: if idx == 2 { 90.0 } else { -100.0 - cost },
                });
            },
        )
    };
    assert!(
        outcomes.iter().position(Result::is_err).is_some(),
        "the plateau must eventually close the window: {outcomes:?}",
    );
    let published = published.expect("the closed window must publish an incumbent");
    assert_eq!(
        published.value, -100.0,
        "the incumbent must be the ACCEPTED trial, not the last rescue poke: {published:?}",
    );
    assert_eq!(
        published.rho,
        array![11.0],
        "and its ρ must travel with it: {published:?}",
    );
}

// ─── #2613 the stationarity band's anchor ─────────────────────────────────────

/// #2613 — the band the SOLVER is judged against must not move when the search
/// is started somewhere else.
///
/// `opt` resolves a `GradientTolerance` once, at run start, against the seed
/// cost. Delegating a `rel_cost` component therefore made `1 + |V(seed)|` the
/// anchor, and on #2392's exponentially stiff recovery that spread the band
/// over eighteen orders across the seeds of ONE fit: a lattice seed at ρ = 1.0,
/// where the criterion is 1.79e13, produced `|g| < 1.792397e8` — a threshold no
/// gradient can fail — and the solver claimed convergence on the wrong rail.
#[test]
fn solver_stationarity_band_is_seed_invariant_2613() {
    let config = OuterConfig {
        tolerance: 1.0e-5,
        objective_scale: Some(1_000.0),
        ..OuterConfig::default()
    };
    let band = outer_gradient_tolerance(&config);
    assert!(
        band.rel_cost.is_none(),
        "a cost-relative component is resolved by opt against the SEED cost, which is \
         precisely the anchor #2613 removes",
    );
    assert!(
        band.rel_initial_grad.is_none(),
        "likewise for a seed-gradient-relative component",
    );
    // The #2392 lattice seed and the optimum it is trying to reach, seventeen
    // orders apart in criterion value.
    let at_lattice_seed = band.threshold(1.792_396_548_924e13, 3.584_853e13);
    let at_optimum = band.threshold(-5.0e3, 0.0);
    assert_eq!(
        at_lattice_seed.to_bits(),
        at_optimum.to_bits(),
        "the solver band moved with the seed: {at_lattice_seed:.6e} vs {at_optimum:.6e}",
    );
    assert!(
        at_lattice_seed < 1.0,
        "a stationarity threshold of {at_lattice_seed:.6e} is not a stationarity test",
    );
}

/// #2613 — and the CERTIFICATE keeps the mgcv `magic` rule it always meant:
/// `‖g‖ ≤ τ·(1 + |V|)` anchored at the criterion value of the point being
/// judged. That anchor was never wrong; it is resolved per point, so anchoring
/// it right costs nothing. Naming the two bands separately is what makes the
/// asymmetry impossible to reintroduce.
#[test]
fn certificate_stationarity_band_still_tracks_the_judged_point_2613() {
    let config = OuterConfig {
        tolerance: 1.0e-5,
        objective_scale: Some(1_000.0),
        ..OuterConfig::default()
    };
    let at_optimum = outer_stationarity_band_at(&config, -5.0e3);
    let at_a_far_worse_point = outer_stationarity_band_at(&config, -5.0e6);
    assert!(
        at_a_far_worse_point > at_optimum,
        "the certificate band must scale with the criterion at the point it judges: \
         {at_a_far_worse_point:.6e} vs {at_optimum:.6e}",
    );
    // 1e-5 · (1 + 5000), the value the #2392 recovery certifies against.
    assert!(
        (at_optimum - 5.001e-2).abs() <= 1.0e-9,
        "certificate band at V = -5e3 should be τ·(1+|V|) = 5.001e-2, got {at_optimum:.9e}",
    );
    // A non-finite criterion licenses no score-relative widening at all: the
    // band falls back to the solver's, which is where the invariant below
    // starts.
    assert_eq!(
        outer_stationarity_band_at(&config, f64::NAN).to_bits(),
        outer_gradient_tolerance(&config).abs.to_bits(),
        "a non-finite cost must fall back to the declared band, never widen",
    );
}

/// #2613 — the certificate's band is never STRICTER than the band the solver
/// was told to reach.
///
/// A certificate tighter than the solver's threshold manufactures the "solver
/// claimed convergence, certificate refused" family out of nothing but a
/// disagreement between two spellings of one tolerance: the optimizer stops
/// exactly where it was asked to and is then told the stop was illegitimate.
/// Looser is fine — that is what the score-relative widening is for. The sweep
/// below covers the regime where the point's own criterion is SMALLER than the
/// declared scale, which is the only way the two can cross.
#[test]
fn certificate_band_never_undercuts_the_solver_band_2613() {
    for scale in [None, Some(1.0), Some(80.0), Some(1_000.0), Some(1.0e6)] {
        for tolerance in [1.0e-8, 1.0e-5, 1.0e-3] {
            let config = OuterConfig {
                tolerance,
                objective_scale: scale,
                ..OuterConfig::default()
            };
            let solver_band = outer_gradient_tolerance(&config).abs;
            for cost in [
                0.0,
                -1.0e-9,
                1.0,
                -80.0,
                1.0e3,
                -5.0e3,
                1.0e12,
                f64::NAN,
                f64::INFINITY,
            ] {
                let certificate_band = outer_stationarity_band_at(&config, cost);
                assert!(
                    certificate_band >= solver_band,
                    "certificate band {certificate_band:.6e} undercuts the solver band                      {solver_band:.6e} at scale={scale:?} tolerance={tolerance:.0e}                      cost={cost:.3e}",
                );
                assert!(
                    certificate_band.is_finite() && certificate_band > 0.0,
                    "a stationarity bound must be a usable positive number, got                      {certificate_band:?} at scale={scale:?} tolerance={tolerance:.0e}                      cost={cost:.3e}",
                );
            }
        }
    }
}

/// #2613 — moving the anchor from the seed to the declared scale is
/// MAGNITUDE-PRESERVING on the routes that declare one, which is why it is not
/// a tightening in disguise.
///
/// A REML/LAML score is a sum over `n` rows, so `1 + |V| = O(n)` is exactly
/// what `1 + objective_scale` says when the scale is `n_obs`. The declared band
/// must land on the same order as the point-anchored band evaluated at a
/// criterion of that size.
#[test]
fn declared_objective_scale_preserves_the_score_relative_magnitude_2613() {
    let n_obs = 1_000.0;
    let config = OuterConfig {
        tolerance: 1.0e-5,
        objective_scale: Some(n_obs),
        ..OuterConfig::default()
    };
    let declared = outer_gradient_tolerance(&config).abs;
    let at_a_score_of_that_size = outer_stationarity_band_at(&config, -n_obs);
    let ratio = declared / at_a_score_of_that_size;
    assert!(
        (0.5..=2.0).contains(&ratio),
        "the declared band {declared:.6e} and the point-anchored band \
         {at_a_score_of_that_size:.6e} at |V| = n must agree to within a factor of two \
         (ratio {ratio:.3})",
    );
    // Without a declared scale gam does not know the criterion's magnitude, and
    // says so by falling back to the absolute tolerance rather than
    // substituting a trajectory point for it. The cost-stall guard's
    // `flat_valley_converged_grad_bound(best_value)` — anchored at the BEST
    // iterate, i.e. the correctly-anchored version of the same idea — is what
    // covers that case.
    let undeclared = OuterConfig {
        tolerance: 1.0e-5,
        objective_scale: None,
        ..OuterConfig::default()
    };
    assert_eq!(
        outer_gradient_tolerance(&undeclared).abs.to_bits(),
        1.0e-5_f64.to_bits(),
        "an undeclared route must get its own absolute tolerance, not a guess",
    );
}

// ─── #2458 the derived standard, and the typed inability to reach it ─────────

/// One second-order-stationary point, certified twice: once by a route that
/// declares a Dense analytic Hessian, once by a route that declares none.
///
/// `V(ρ) = ½ρ²` at `ρ = 2e-6` with `objective_scale = 80`. The arithmetic
/// gradient floor is `80·√ε = 1.19e-6`, so `|Pg| = 2e-6` REFUSES on the raw
/// band — while the Newton decrement `½·(2e-6)² = 2e-12` is orders below any
/// outer objective tolerance, i.e. the point is stationary to second order and
/// the curvature-resolvability rung is exactly what exists to say so.
///
/// `declares_hessian` is the ONLY difference between the two calls.
fn certify_quadratic_at_declared_curvature_2458(
    declares_hessian: bool,
    theta: f64,
    search_iterations: usize,
) -> Result<OuterCriterionCertificate, EstimationError> {
    let config = OuterConfig {
        tolerance: 1.0e-12,
        objective_scale: Some(80.0),
        ..OuterConfig::default()
    };
    let mut obj = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(if declares_hessian {
            DeclaredHessianForm::Dense
        } else {
            DeclaredHessianForm::Unavailable
        })
        .build_objective(
            (),
            move |_: &mut (), rho: &Array1<f64>| Ok(0.5 * rho[0] * rho[0]),
            move |_: &mut (), rho: &Array1<f64>| {
                Ok(OuterEval {
                    cost: 0.5 * rho[0] * rho[0],
                    gradient: array![rho[0]],
                    hessian: if declares_hessian {
                        HessianValue::Dense(array![[1.0]])
                    } else {
                        HessianValue::Unavailable
                    },
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
    let mut result = OuterResult::new(
        array![theta],
        0.5 * theta * theta,
        search_iterations,
        true,
        OuterPlan {
            solver: Solver::Bfgs,
            hessian_source: HessianSource::BfgsApprox,
        },
    );
    certify_outer_optimality(&mut obj, &config, "curvature-rung-2458", &mut result)
}

/// #2458 — a route that supplies exact curvature reaches the derived standard,
/// and one that supplies none is REFUSED and says so in its rung.
///
/// The two certificates below judge the SAME point with the SAME criterion and
/// differ only in `DeclaredHessianForm`. That is deliberately still a
/// difference in outcome, and this test pins which difference is acceptable.
///
/// The rejected repair was to close the gap inside the certifier by
/// forward-differencing the gradient-only route's analytic gradient. It closed
/// the gap and it was wrong twice over: SPEC line 2 permits finite differences
/// only outside production, and the distinction that makes that rule bite here
/// is that the workspace's other production finite difference (the ψ audit in
/// `run_plan.rs`, behind `outer_gradient_fd_capture_enabled`) is read by nothing
/// outside tests — it RECORDS what happened, while a rung that overwrites
/// `stationarity_bound` DECIDES what is true.
///
/// So the contract is: the derived standard is reached by supplying curvature,
/// never by estimating it on the route's behalf, and a route that cannot supply
/// it is refused **legibly** — `derived_standard = false` on the rung — rather
/// than being handed the same label on weaker evidence. The fix for a route
/// that deserves the standard is to give it a real second derivative, which is
/// what the constant-curvature κ profile now does.
#[test]
fn exact_curvature_reaches_the_derived_standard_and_its_absence_is_recorded_2458() {
    let theta = 2.0e-6;
    let with_hessian = certify_quadratic_at_declared_curvature_2458(true, theta, 1)
        .expect("the declared-Hessian route certifies via the curvature rung");

    assert!(with_hessian.certifies(), "{}", with_hessian.summary());
    assert_eq!(
        with_hessian.stationarity.rung().label,
        "curvature-resolvability",
        "the declared-Hessian route's rung changed: {}",
        with_hessian.summary(),
    );
    assert!(
        with_hessian.stationarity.rung().derived_standard,
        "the decrement test IS the derived standard: {}",
        with_hessian.summary(),
    );

    // The test proves nothing about the widening unless the raw band really
    // would have refused this point.
    let arithmetic_floor = 80.0 * f64::EPSILON.sqrt();
    assert!(
        with_hessian.stationarity.projected_norm() > arithmetic_floor,
        "|Pg| must sit ABOVE the un-widened band, else no widening was needed: {}",
        with_hessian.summary(),
    );

    // Same point, same criterion, no declared curvature: refused, and the
    // refusal names a rung that does NOT claim to be the derived standard.
    let refusal = certify_quadratic_at_declared_curvature_2458(false, theta, 1)
        .expect_err("a route with no curvature cannot run the decrement test");
    let message = refusal.to_string();
    assert!(
        message.contains("NOT STATIONARY"),
        "the refusal must be the ordinary non-stationarity one: {message}",
    );
    assert!(
        message.contains("derived_standard=false"),
        "a refusal on a rung that is not the derived standard must SAY so, which is \
         what makes the remaining gap attributable instead of invisible: {message}",
    );
    assert!(
        !message.contains("fd-gradient"),
        "no rung may be produced by finite-differencing a gradient in production \
         (SPEC line 2): {message}",
    );
}

/// #2458 — the curvature rung is a genuine test, not a blanket loosening.
///
/// The widened bound is `|Pg|·√(τ/Δpred)`, which exceeds `|Pg|` **iff**
/// `Δpred ≤ τ`. At a point with real available descent `Δpred ≫ τ`, so the rung
/// still wins the ladder's max — it is the largest available bound — and still
/// lands orders BELOW the gradient it is judging. Widening is not rescuing.
#[test]
fn the_curvature_rung_still_refuses_genuine_nonstationarity_2458() {
    // |Pg| = 1 against a unit Hessian gives Δpred = 0.5 against
    // τ = 1e-7·(1+0.5) = 1.5e-7, so the bound is √(3e-7) = 5.477e-4: the widest
    // rung on the ladder, and 1826x below the gradient.
    let refusal = certify_quadratic_at_declared_curvature_2458(true, 1.0, 1)
        .expect_err("a point with a half-unit predicted decrease is not stationary");
    let message = refusal.to_string();
    assert!(
        message.contains("NOT STATIONARY"),
        "the refusal must be the ordinary non-stationarity one: {message}",
    );
    assert!(
        message.contains("bound=5.477e-4"),
        "the widened bound must be the decrement test's own answer, not a rescue: {message}",
    );
    // The point of the assertion: three orders of margin between the widest
    // bound the ladder can produce here and the gradient it is judging.
    assert!(
        message.contains("|Pg|=1.000e0 > bound=5.477e-4"),
        "the refusal must compare the two directly: {message}",
    );
}

/// #2458 — exactly one rung may call itself the derived standard, and it is the
/// one computed from the family's own exact curvature.
///
/// This is the invariant the reopened issue turned on. A second rung carrying
/// `derived_standard = true` on `O(√ε)` evidence is how "which derivative
/// machinery does this route implement" gets back into the answer to "which
/// standard was this fit held to" — the exact substitution #2458 exists to
/// remove. Enumerated rather than spot-checked so a new rung cannot claim the
/// flag by being added below the ones a spot check happened to name.
#[test]
fn exactly_one_rung_is_the_derived_standard_2458() {
    let rungs = [
        StationarityBoundSource::SolverBand,
        StationarityBoundSource::CertificateScoreRelative,
        StationarityBoundSource::ProbeNoiseFloor,
        StationarityBoundSource::CurvatureResolvability,
        StationarityBoundSource::GradientReproducibility,
        StationarityBoundSource::FixedPointResidual,
        StationarityBoundSource::CallerRequirement,
    ];
    let derived: Vec<&'static str> = rungs
        .iter()
        .filter(|rung| rung.is_derived_standard())
        .map(|rung| rung.label())
        .collect();
    assert_eq!(
        derived,
        vec!["curvature-resolvability"],
        "the derived standard must be the exact-curvature decrement test alone",
    );
    // And the enumeration above must be the whole enum: a rung added without
    // being listed here would slip past the assertion. `label()` is total over
    // the enum, so a distinct label per listed variant plus a count check is the
    // available proof that nothing was dropped from the list.
    let mut labels: Vec<&'static str> = rungs.iter().map(|rung| rung.label()).collect();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(
        labels.len(),
        rungs.len(),
        "two rungs share a label, so the enumeration above is not what it appears to be",
    );
    // A distinct-label count still cannot see a variant that was never listed,
    // so make the compiler the gate (#2688 added one and this list had to be
    // edited by hand): the binding below names every variant and no wildcard,
    // so it is irrefutable only while this list is the whole enum. The next
    // variant added makes it refutable, and a refutable `let` is a compile
    // error HERE, on the test that owns the "which rungs exist" question.
    for rung in rungs {
        let (StationarityBoundSource::SolverBand
        | StationarityBoundSource::CertificateScoreRelative
        | StationarityBoundSource::ProbeNoiseFloor
        | StationarityBoundSource::CurvatureResolvability
        | StationarityBoundSource::GradientReproducibility
        | StationarityBoundSource::FixedPointResidual
        | StationarityBoundSource::CallerRequirement) = rung;
    }
}
