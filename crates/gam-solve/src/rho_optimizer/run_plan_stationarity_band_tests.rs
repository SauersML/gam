// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): the outer stationarity band — the #2613 gradient-only stiff-ridge
// replay, the cost-stall window counting accepted steps, the band's anchor,
// and the #2458 derived standard on routes with no declared curvature. Scope
// comes from the parent via `use super::*`; the split is purely physical.

use super::*;
use ndarray::array;

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
    // The #2392 fixture's face, stated for the audit and the recovery alike: an
    // undeclared outer domain is the supported log-strength domain (04726d916),
    // where 29.9 is interior.
    const WRONG_RAIL_FACE: f64 = 30.0;

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
        log::trace!("zz_measure #2613: a logger was already installed by another test");
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
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(
            Array1::from_elem(1, -WRONG_RAIL_FACE),
            Array1::from_elem(1, WRONG_RAIL_FACE),
        );
    let audit_cost = cost.clone();
    let audit_eval = eval.clone();
    let mut audit_obj = audit_problem.build_objective(
        (),
        move |_: &mut (), rho: &Array1<f64>| Ok(audit_cost(rho)),
        move |_: &mut (), rho: &Array1<f64>| Ok(audit_eval(rho)),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let refusal = audit_stationary_point_in(
        &mut audit_obj,
        audit_problem.config(),
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
        .with_bounds(
            Array1::from_elem(1, -WRONG_RAIL_FACE),
            Array1::from_elem(1, WRONG_RAIL_FACE),
        )
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
    let mut guard = CostStallGuard::new(1.0e-7, COST_STALL_WINDOW, &claim_band_config(1.0e-3), exit.clone());
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
        stratum_rank: None,
        stratum_probe: None,
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
        problem_size: problem_size_2954(1_000, 10),
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

/// A declared problem size: the formation count the Theorem 9 rounding band
/// is charged at.
fn problem_size_2954(n_obs: usize, p_coefficients: usize) -> crate::rho_optimizer::OuterProblemSize {
    crate::rho_optimizer::OuterProblemSize {
        n_obs: Some(n_obs),
        p_coefficients: Some(p_coefficients),
    }
}

/// Published gradient parts, one per `(rank, fixed_beta, logdet_h, total)`,
/// with `logdet_s = −½·rank` as the REML engine forms it; whatever `total`
/// leaves over lands in the KKT channel exactly as it does in production.
fn evidence_2954(
    coordinates: &[(usize, f64, f64, f64)],
) -> crate::estimate::outer_eval_capture::CertificateEvidence {
    crate::estimate::outer_eval_capture::CertificateEvidence {
        parts: coordinates
            .iter()
            .enumerate()
            .map(|(index, &(rank, fixed_beta, logdet_h, total))| {
                crate::estimate::outer_eval_capture::RhoGradientParts {
                    index,
                    lambda: 1.0,
                    block_quadratic: 2.0 * fixed_beta,
                    rank,
                    dim: rank,
                    fixed_beta,
                    logdet_h,
                    frozen_logdet_h: logdet_h,
                    mode_response_logdet_h: 0.0,
                    logdet_s: -0.5 * rank as f64,
                    total,
                }
            })
            .collect(),
        ..Default::default()
    }
}

/// #2954 — the certificate band is not a function of the criterion's value.
///
/// Until #2954 the certificate widened to `τ·(1 + |V|)` at the judged point
/// (#2613). A REML/LAML score is a sum over rows, so that band grew with `n`
/// and with any additive constant in `V`, while the ρ-gradient it judged is a
/// difference of O(rank) terms. The replacement reads the gradient's own parts
/// and nothing else: the same parts under criteria seventeen orders apart give
/// the same band to the bit.
#[test]
fn the_certificate_band_does_not_track_the_criterion_value_2954() {
    let config = OuterConfig {
        tolerance: 1.0e-5,
        problem_size: problem_size_2954(1_000, 10),
        ..OuterConfig::default()
    };
    let gradient = array![2.0e-6, -3.0e-6];
    let band_under = |cost: f64| {
        let mut evidence = evidence_2954(&[(4, 0.8, 1.2, 2.0e-6), (9, 3.5, 1.0, -3.0e-6)]);
        evidence.criterion = Some(crate::estimate::outer_eval_capture::CertificateCriterion {
            cost,
            fixed_beta: cost,
            logdet_h: 0.0,
            logdet_s: 0.0,
            kkt: 0.0,
            inner_residual_energy: None,
        });
        outer_certificate_band_at(&config, &gradient, &evidence)
    };
    let at_optimum = band_under(-5.0e3);
    for cost in [-5.0e6, 1.792_396_548_924e13, 0.0] {
        let elsewhere = band_under(cost);
        assert_eq!(
            elsewhere.bound.to_bits(),
            at_optimum.bound.to_bits(),
            "the band moved with the criterion value: {:.6e} at V={cost:.3e} vs {:.6e}",
            elsewhere.bound,
            at_optimum.bound,
        );
        assert_eq!(elsewhere.source, StationarityBoundSource::CoordinateBand);
    }
}

/// #2954 — the per-coordinate band does not grow with the row count.
///
/// The same gradient parts judged at `n` from two thousand to twenty million
/// rows: the only `n` dependence is the rounding charge `ε_j = γ_m·Σ|parts|`
/// with `m = n + p²`, which SHRINKS the band. The removed floor was `n·√ε`
/// (7.3e-6 at the #2954 fixture's `n = 490`) and the removed declared-scale
/// band `τ·(1 + n)` (6.0 at `n = 300,000`, where a seed certified in zero
/// iterations).
#[test]
fn the_coordinate_band_is_invariant_to_the_row_count_2954() {
    let tolerance = 1.0e-5;
    let gradient = array![1.0e-6];
    let evidence = evidence_2954(&[(8, 1.5, 2.25, 1.0e-6)]);
    // τ = tolerance·(1 + ½·8 + 1.5).
    let tau = tolerance * (1.0 + 4.0 + 1.5);
    let bands: Vec<f64> = [2_000, 20_000, 200_000, 300_000, 2_000_000, 20_000_000]
        .into_iter()
        .map(|n_obs| {
            let config = OuterConfig {
                tolerance,
                problem_size: problem_size_2954(n_obs, 12),
                ..OuterConfig::default()
            };
            let band = outer_certificate_band_at(&config, &gradient, &evidence);
            assert_eq!(band.source, StationarityBoundSource::CoordinateBand);
            assert!(
                band.bound <= tau && band.bound > 0.99 * tau,
                "n={n_obs}: band {:.6e} must be τ={tau:.6e} less a rounding charge far \
                 below it",
                band.bound,
            );
            band.bound
        })
        .collect();
    assert!(
        bands.windows(2).all(|pair| pair[1] <= pair[0]),
        "more rows can only charge more rounding, never widen the band: {bands:?}",
    );
}

/// #2954 — Theorem 9. The scalar gauge the certificate compares `‖Pĝ‖₂`
/// against passes exactly when every coordinate clears its own band, and a
/// pass on a resolvable coordinate proves the EXACT component within `τ_j`:
/// `|ĝ_j| ≤ τ_j − ε_j` and `|g_j − ĝ_j| ≤ ε_j`.
#[test]
fn a_coordinate_band_pass_proves_the_exact_component_within_tau_2954() {
    let config = OuterConfig {
        tolerance: 1.0e-6,
        problem_size: problem_size_2954(5_000, 20),
        ..OuterConfig::default()
    };
    let evidence = evidence_2954(&[(2, 0.1, 0.9, 0.0), (30, 12.0, 3.0, 0.0)]);
    let bands = outer_coordinate_bands(&config, 2, &evidence)
        .expect("a declared size charges rounding")
        .into_iter()
        .map(|band| band.expect("both coordinates published parts"))
        .collect::<Vec<_>>();
    for band in &bands {
        assert!(!band.is_arithmetic_limited(), "fixture must be resolvable: {band:?}");
    }
    let mut passes = 0;
    let mut refusals = 0;
    for a in [0.0, 0.5, 0.99, 1.01, 3.0] {
        for b in [0.0, 0.25, 0.98, 1.02, 10.0] {
            let gradient = array![a * bands[0].band(), -b * bands[1].band()];
            let got = outer_certificate_band_at(&config, &gradient, &evidence);
            let norm = gradient.iter().map(|v| v * v).sum::<f64>().sqrt();
            let every_coordinate_clears = a <= 1.0 && b <= 1.0;
            assert_eq!(
                norm <= got.bound,
                every_coordinate_clears,
                "scalar gauge disagrees with the per-coordinate test at ({a}, {b}): \
                 ‖Pĝ‖={norm:.6e}, bound={:.6e}",
                got.bound,
            );
            if every_coordinate_clears {
                passes += 1;
                for (component, band) in gradient.iter().zip(&bands) {
                    assert!(
                        component.abs() + band.epsilon <= band.tau,
                        "a pass must leave room for the rounding: |ĝ|={:.6e}, ε={:.6e}, \
                         τ={:.6e}",
                        component.abs(),
                        band.epsilon,
                        band.tau,
                    );
                }
            } else {
                refusals += 1;
            }
        }
    }
    assert!(passes > 0 && refusals > 0, "the sweep must exercise both verdicts");
}

/// #2954 — a coordinate whose rounding exceeds half its resolution is
/// certified at the arithmetic's own resolution and SAYS so.
///
/// Cancelling log-determinant channels of `±1e10` against a rank-4 penalty
/// charge `ε ≈ γ_m·2e10`, beyond `τ/2`. No computed component proves
/// `|g| ≤ τ` there. The band is `ε` (a pass proves `|g| ≤ 2ε`), the label is
/// `arithmetic-limited`, and it is not the derived standard; a component above
/// `ε` is refused like any other.
#[test]
fn an_arithmetic_limited_coordinate_is_labelled_2954() {
    let config = OuterConfig {
        tolerance: 1.0e-5,
        problem_size: problem_size_2954(1_000, 10),
        ..OuterConfig::default()
    };
    let evidence = evidence_2954(&[(4, 1.0, 1.0e10, 0.0)]);
    let band = outer_coordinate_bands(&config, 1, &evidence).unwrap()[0].unwrap();
    assert!(band.is_arithmetic_limited(), "fixture must be arithmetic-limited: {band:?}");
    assert_eq!(band.band(), band.epsilon);

    let inside = outer_certificate_band_at(&config, &array![0.5 * band.epsilon], &evidence);
    assert_eq!(inside.source, StationarityBoundSource::ArithmeticLimited);
    assert_eq!(inside.bound, band.epsilon);
    assert!(!inside.source.is_derived_standard());

    let beyond = outer_certificate_band_at(&config, &array![3.0 * band.epsilon], &evidence);
    assert_eq!(beyond.source, StationarityBoundSource::ArithmeticLimited);
    assert!(
        3.0 * band.epsilon > beyond.bound,
        "a component three rounding bands out is not stationary at any resolution",
    );

    // Beside a resolvable coordinate that binds, the verdict is the resolvable
    // one's and so is the label.
    let mixed = evidence_2954(&[(4, 1.0, 1.0e10, 0.0), (8, 1.5, 2.25, 0.0)]);
    let bands = outer_coordinate_bands(&config, 2, &mixed).unwrap();
    let resolvable = bands[1].unwrap();
    let got = outer_certificate_band_at(
        &config,
        &array![0.1 * band.epsilon, 2.0 * resolvable.band()],
        &mixed,
    );
    assert_eq!(got.source, StationarityBoundSource::CoordinateBand);
}

/// #2954 — edge cases of the scalar gauge: a coordinate with no published
/// parts is held to the absolute tolerance, `Pĝ = 0` reports the smallest
/// band, and a projected-out coordinate (`Pĝ_j = 0`) constrains nothing.
#[test]
fn the_coordinate_band_gauge_edge_cases_2954() {
    let tolerance = 1.0e-5;
    let config = OuterConfig {
        tolerance,
        problem_size: problem_size_2954(1_000, 10),
        ..OuterConfig::default()
    };
    // Parts for coordinate 0 only.
    let evidence = evidence_2954(&[(8, 1.5, 2.25, 0.0)]);
    let bands = outer_coordinate_bands(&config, 2, &evidence).unwrap();
    assert!(bands[1].is_none());
    let wide = bands[0].unwrap().band();
    assert!(wide > tolerance);

    // Coordinate 1 binds at the absolute tolerance.
    let got = outer_certificate_band_at(&config, &array![0.0, 2.0 * tolerance], &evidence);
    assert_eq!(got.bound, tolerance);

    // Pĝ = 0: the smallest band, which here is the un-parted coordinate's.
    let zero = outer_certificate_band_at(&config, &array![0.0, 0.0], &evidence);
    assert_eq!(zero.bound, tolerance);

    // Only coordinate 0 carries gradient: its own band, whatever coordinate 1
    // would have demanded.
    let only_first = outer_certificate_band_at(&config, &array![0.5 * wide, 0.0], &evidence);
    assert!((only_first.bound - wide).abs() <= 1.0e-15 * wide);
    assert_eq!(only_first.source, StationarityBoundSource::CoordinateBand);

    // No declared size: nothing to charge rounding at, so the declared band.
    let no_size = OuterConfig {
        tolerance,
        ..OuterConfig::default()
    };
    let got = outer_certificate_band_at(&no_size, &array![0.5 * wide, 0.0], &evidence);
    assert_eq!(got.source, StationarityBoundSource::SolverBand);
    assert_eq!(got.bound, tolerance);
}

/// #2954 — the certificate's band is never stricter than the band the solver
/// was told to reach wherever the arithmetic resolves that band.
///
/// A certificate tighter than the solver's threshold manufactures the "solver
/// claimed convergence, certificate refused" family out of nothing but a
/// disagreement between two spellings of one tolerance (#2613). With parts,
/// the band is `τ_j − ε_j = tolerance + (tolerance·s_j − ε_j)`, so it clears
/// the solver's band exactly when the rounding stays below `tolerance·s_j`.
#[test]
fn certificate_band_never_undercuts_a_resolvable_solver_band_2954() {
    let mut resolvable_cases = 0;
    for tolerance in [1.0e-8, 1.0e-5, 1.0e-3] {
        for n_obs in [100, 10_000, 1_000_000] {
            let config = OuterConfig {
                tolerance,
                problem_size: problem_size_2954(n_obs, 30),
                ..OuterConfig::default()
            };
            let solver_band = outer_gradient_tolerance(&config).abs;
            assert_eq!(solver_band, tolerance);
            for (rank, fixed_beta, logdet_h) in
                [(1, 0.0, 0.4), (4, 0.5, 1.5), (40, 30.0, 12.0), (2, 1.0e-3, 0.999)]
            {
                let evidence = evidence_2954(&[(rank, fixed_beta, logdet_h, 1.0e-9)]);
                let band = outer_coordinate_bands(&config, 1, &evidence).unwrap()[0].unwrap();
                let got = outer_certificate_band_at(&config, &array![1.0e-9], &evidence);
                assert!(
                    got.bound.is_finite() && got.bound > 0.0,
                    "a stationarity bound must be a usable positive number, got {:?}",
                    got.bound,
                );
                let scale = 0.5 * rank as f64 + fixed_beta;
                if band.epsilon <= tolerance * scale {
                    resolvable_cases += 1;
                    assert!(
                        got.bound >= solver_band,
                        "certificate band {:.6e} undercuts the solver band {solver_band:.6e} \
                         at tolerance={tolerance:.0e} n={n_obs} rank={rank}",
                        got.bound,
                    );
                }
            }
            // Without parts the certificate applies the solver's band itself.
            let bare = outer_certificate_band_at(
                &config,
                &array![1.0e-9],
                &crate::estimate::outer_eval_capture::CertificateEvidence::default(),
            );
            assert_eq!(bare.bound.to_bits(), solver_band.to_bits());
        }
    }
    assert!(resolvable_cases > 0, "the sweep must exercise the resolvable regime");
}

// ─── #2458 the derived standard, and the typed inability to reach it ─────────

/// One second-order-stationary point, certified twice: once by a route that
/// declares a Dense analytic Hessian, once by a route that declares none.
///
/// `V(ρ) = ½ρ²` at `ρ = 2e-6` with `tolerance = 1e-12`. The route publishes no
/// gradient parts, so the raw band is the tolerance and `|Pg| = 2e-6` REFUSES
/// on it — while the Newton decrement `½·(2e-6)² = 2e-12` is orders below any
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
/// only in tests, and a rung that overwrites `stationarity_bound` DECIDES what
/// is true rather than recording what happened. Production now differences
/// nothing at all: the ψ audit `run_plan.rs` once ran at an armed seed is an
/// analytic seed probe whose test forms its own difference (#2901).
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
    let raw_band = 1.0e-12;
    assert!(
        with_hessian.stationarity.projected_norm() > raw_band,
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

/// #2458 — only the rungs computed from the family's own exact curvature may
/// call themselves the derived standard.
///
/// This is the invariant the reopened issue turned on. A rung carrying
/// `derived_standard = true` on `O(√ε)` evidence is how "which derivative
/// machinery does this route implement" gets back into the answer to "which
/// standard was this fit held to" — the exact substitution #2458 exists to
/// remove. Enumerated rather than spot-checked so a new rung cannot claim the
/// flag by being added below the ones a spot check happened to name.
///
/// #2954 added the Newton-decrement verdict on rounding bands, the same exact
/// curvature judged against the arithmetic's resolution instead of an n-scaled
/// tolerance, in its deciding and its undecided form. The typed refusals of its
/// polish (#2954, #3012) are not stationarity standards and do not claim it.
/// The per-coordinate Theorem 9 band is: it is derived from the gradient's own
/// scale and rounding, and a pass proves the exact component within the
/// requested tolerance. Its arithmetic-limited form proves only `2ε_j` and does
/// not claim it.
#[test]
fn only_exact_curvature_rungs_are_the_derived_standard_2458() {
    let rungs = [
        StationarityBoundSource::SolverBand,
        StationarityBoundSource::CoordinateBand,
        StationarityBoundSource::ArithmeticLimited,
        StationarityBoundSource::CurvatureResolvability,
        StationarityBoundSource::GradientReproducibility,
        StationarityBoundSource::FixedPointResidual,
        StationarityBoundSource::CallerRequirement,
        StationarityBoundSource::NewtonDecrement,
        StationarityBoundSource::NewtonDecrementUndecided,
        StationarityBoundSource::RepresentabilityFace,
        StationarityBoundSource::NewtonBacktrackUnresolved,
    ];
    let derived: Vec<&'static str> = rungs
        .iter()
        .filter(|rung| rung.is_derived_standard())
        .map(|rung| rung.label())
        .collect();
    assert_eq!(
        derived,
        vec![
            "coordinate-band",
            "curvature-resolvability",
            "newton-decrement",
            "newton-decrement-undecided"
        ],
        "the derived standard must be the exact-curvature decrement test and the \
         Theorem 9 band on the gradient's own parts alone",
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
        | StationarityBoundSource::CoordinateBand
        | StationarityBoundSource::ArithmeticLimited
        | StationarityBoundSource::CurvatureResolvability
        | StationarityBoundSource::GradientReproducibility
        | StationarityBoundSource::FixedPointResidual
        | StationarityBoundSource::CallerRequirement
        | StationarityBoundSource::NewtonDecrement
        | StationarityBoundSource::NewtonDecrementUndecided
        | StationarityBoundSource::RepresentabilityFace
        | StationarityBoundSource::NewtonBacktrackUnresolved) = rung;
    }
}
