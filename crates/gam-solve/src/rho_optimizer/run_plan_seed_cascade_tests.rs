// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): the seed cascade itself — keep-best / parsimony ranking, Gaussian
// multistart, expensive-seed screening and its cap ladder, the effective seed
// budget, and seed projection before the validation eval. Child modules see
// the parent's entire scope (helpers AND imports) via `use super::*`, so the
// split is purely physical.

use super::*;
use ndarray::array;

/// Keep-best must rank CERTIFICATION above VALUE, the invariant `run_plan.rs`
/// relies on when it routes the #1371/#1476 ARC box-corner substitution through
/// keep-best "as a NON-converged candidate" so an earlier converged seed still
/// wins. The costs below are that scenario's own numbers: a genuine interior
/// optimum at ~133 and a degenerate stall caching a spuriously LOWER ~65.
#[test]
fn keep_best_ranks_convergence_above_value() {
    let plan = OuterPlan {
        solver: Solver::Bfgs,
        hessian_source: HessianSource::BfgsApprox,
    };
    let converged = OuterResult::new(array![-1.0], 133.0, 12, true, plan);
    let stalled_cheaper = OuterResult::new(array![8.0], 65.0, 58, false, plan);

    assert!(
        !candidate_improves_best(&stalled_cheaper, Some(&converged)),
        "a non-converged candidate must not displace a converged best, however \
         much lower its cached cost is — that cost is where the search stopped, \
         not a claim about an optimum"
    );
    assert!(
        candidate_improves_best(&converged, Some(&stalled_cheaper)),
        "a converged candidate must displace a non-converged best even when its \
         value is worse"
    );

    // With convergence equal on both sides, value decides exactly as before.
    let cheaper_converged = OuterResult::new(array![-2.0], 132.0, 9, true, plan);
    assert!(candidate_improves_best(&cheaper_converged, Some(&converged)));
    assert!(!candidate_improves_best(&converged, Some(&cheaper_converged)));
    let dearer_stall = OuterResult::new(array![9.0], 70.0, 4, false, plan);
    assert!(candidate_improves_best(&stalled_cheaper, Some(&dearer_stall)));

    // Nothing to beat: the first candidate is always adopted, converged or not,
    // so a single-start fit still returns its result unchanged.
    assert!(candidate_improves_best(&stalled_cheaper, None));
}

#[test]
fn parsimonious_keep_best_breaks_laml_tie_toward_more_smoothing() {
    let plan = OuterPlan {
        solver: Solver::Bfgs,
        hessian_source: HessianSource::BfgsApprox,
    };
    let rho_dim = 2usize;

    // Two CONVERGED optima whose LAML values are a statistical tie (within the
    // relative band): a flexible (low-Σρ) basin scoring epsilon BETTER, and a
    // parsimonious (high-Σρ) basin. The parsimonious one must win the tie.
    let flexible = OuterResult::new(array![-3.0, -3.0], 100.0, 1, true, plan);
    let mut parsimonious = OuterResult::new(array![3.0, 3.0], 100.05, 1, true, plan);
    parsimonious.final_grad_norm = Some(0.0);

    // gap 0.05 <= 1e-3 * 100.05 (=0.10005) → tie band → prefer larger Σρ.
    assert!(candidate_improves_best_parsimonious(
        &parsimonious,
        Some(&flexible),
        rho_dim,
    ));
    // The flexible (lower-LAML) candidate must NOT displace the parsimonious
    // incumbent on a tie — the tie-break is asymmetric toward more smoothing.
    assert!(!candidate_improves_best_parsimonious(
        &flexible,
        Some(&parsimonious),
        rho_dim,
    ));

    // A DECISIVE LAML advantage for the flexible basin (gap far outside the
    // band) must still win: a fit that genuinely needs the flexibility is not
    // sacrificed to parsimony.
    let decisive_flexible = OuterResult::new(array![-3.0, -3.0], 90.0, 1, true, plan);
    assert!(candidate_improves_best_parsimonious(
        &decisive_flexible,
        Some(&parsimonious),
        rho_dim,
    ));
}

/// #979: the seed budget is a recovery ceiling, not the number of certified
/// basins that parsimony comparison must solve. Once flexible slot 0 and the
/// promoted smoothed slot 1 have both certified, a larger recovery budget must
/// not force an unrelated third outer optimization.
#[test]
fn parsimony_await_stops_after_the_promoted_second_seed() {
    assert!(
        should_await_promoted_parsimony_seed(4, 1, false),
        "slot 0 must await the deliberately promoted slot 1"
    );
    assert!(
        !should_await_promoted_parsimony_seed(4, 2, false),
        "two certified comparison basins exhaust the parsimony contract, even with recovery budget left"
    );
    assert!(
        !should_await_promoted_parsimony_seed(1, 1, false),
        "a single-start request cannot promote a second seed"
    );
    assert!(
        !should_await_promoted_parsimony_seed(4, 1, true),
        "a proven-redundant promoted basin remains waivable"
    );
}

/// #1575: the parsimony-await second-seed waiver fires ONLY for a slot-0 result
/// that is curvature-pinned (score-relative |g| well inside the tie band) AND
/// well-penalized (every leading smoothing λ ≥ 1). Only analytically certified
/// candidates can reach this predicate.
#[test]
fn parsimony_second_seed_waived_only_for_sharp_well_penalized_optimum() {
    let plan = OuterPlan {
        solver: Solver::Arc,
        hessian_source: HessianSource::Analytic,
    };
    let rho_dim = 2usize;
    // `at_band(frac, score)` is the residual gradient sitting `frac`× the
    // score-relative sharpness band: frac<1 is inside (sharp), frac>1 outside.
    let at_band =
        |frac: f64, score: f64| PARSIMONY_SHARP_GRAD_REL_BAND * (1.0 + score.abs()) * frac;

    // The redundant case: converged, every smoothing ρ ≥ 0, residual gradient
    // two orders inside the tie band. Slot 1 would only re-derive this — waive.
    let mut redundant = OuterResult::new(array![3.0, 0.0], 1082.972, 6, true, plan);
    redundant.final_grad_norm = Some(at_band(0.01, 1082.972)); // 0.01× the band → sharp
    assert!(
        parsimony_second_seed_is_redundant(&redundant, rho_dim),
        "a converged, sharp, well-penalized slot-0 optimum makes the heavy seed redundant"
    );
    // Exactly on the band boundary still counts as sharp (≤, not <).
    let mut on_band = redundant.clone();
    on_band.final_grad_norm = Some(at_band(1.0, 1082.972));
    assert!(
        parsimony_second_seed_is_redundant(&on_band, rho_dim),
        "the score-relative band is inclusive at its edge"
    );

    // #1373 under-penalized basin: a smoothing λ < 1 (ρ < 0) is exactly the
    // overshoot the heavy seed guards against — never waive, even when sharp.
    let mut under_penalized = redundant.clone();
    under_penalized.rho = array![-0.5, 3.0];
    assert!(
        !parsimony_second_seed_is_redundant(&under_penalized, rho_dim),
        "a single under-penalized (ρ<0) coordinate keeps the parsimony seed (#1373)"
    );

    // Flat-valley non-sharp optimum: converged at a residual ABOVE the band, so
    // the parsimony tie-break could still slide ρ toward the heavier basin.
    let mut flat_valley = redundant.clone();
    flat_valley.final_grad_norm = Some(at_band(10.0, 1082.972)); // 10× the band → not sharp
    assert!(
        !parsimony_second_seed_is_redundant(&flat_valley, rho_dim),
        "a converged-but-flat optimum above the tie band keeps the parsimony seed"
    );

    // No measured gradient cannot certify sharpness.
    let mut no_grad = redundant.clone();
    no_grad.final_grad_norm = None;
    assert!(
        !parsimony_second_seed_is_redundant(&no_grad, rho_dim),
        "an unmeasured gradient cannot prove a curvature-pinned optimum"
    );

    // The score-relative band scales with |score|: a residual that is absolutely
    // large is still sharp when the LAML magnitude is large enough.
    let mut large_score = OuterResult::new(array![2.0, 2.0], 5.0e6, 6, true, plan);
    large_score.final_grad_norm = Some(40.0); // ≤ 1e-5·(1+5e6) = 50.00001
    assert!(
        parsimony_second_seed_is_redundant(&large_score, rho_dim),
        "sharpness is score-relative, not an absolute gradient threshold"
    );

    // Trailing auxiliary coordinates (e.g. a GAMLSS log-scale predictor) are not
    // smoothing parameters and must not block the waiver: only the leading
    // rho_dim coordinates are tested for λ ≥ 1.
    let mut with_aux = OuterResult::new(array![3.0, -7.0], 100.0, 6, true, plan);
    with_aux.final_grad_norm = Some(at_band(0.1, 100.0));
    assert!(
        parsimony_second_seed_is_redundant(&with_aux, 1),
        "a negative trailing auxiliary coordinate (ρ_dim=1) must not block the waiver"
    );

    // With no smoothing dimension the parsimony tie-break is a no-op.
    assert!(
        !parsimony_second_seed_is_redundant(&redundant, 0),
        "rho_dim=0 has no smoothing parameter for the parsimony seed to decide"
    );
}

#[test]
fn gaussian_multistart_compares_converged_seed_costs() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 2;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let started = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_max_iter(4);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(if theta[0] < -1.0 { 0.0 } else { 10.0 }),
        {
            let started = Arc::clone(&started);
            move |_: &mut (), theta: &Array1<f64>| {
                started.lock().unwrap().push(theta.clone());
                Ok(OuterEval {
                    cost: if theta[0] < -1.0 { 0.0 } else { 10.0 },
                    gradient: array![0.0],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "Gaussian quality multistart")
        .expect("Gaussian multistart should compare both converged seeds");
    let starts = started.lock().unwrap();
    assert!(
        starts.len() >= 2,
        "Gaussian quality mode should not stop at the first converged seed"
    );
    assert!(
        result.rho[0] < -1.0,
        "lower-cost converged Gaussian seed should win"
    );
    assert_eq!(result.final_value, 0.0);
}

/// #1575 end-to-end wiring: drive the real multi-start loop with the
/// parsimonious (GeneralizedLinear) risk profile and `seed_budget = 2`. A slot-0
/// seed that CONVERGES to a sharp, well-penalized optimum (every smoothing
/// λ ≥ 1) must BREAK the multi-start after a single seed — the heavy slot-1 seed
/// is provably redundant. A slot-0 seed that converges to an UNDER-penalized
/// (ρ < 0) optimum is the #1373 overshoot regime, so the heavy seed must STILL
/// run. Counts genuine seed solves by intersecting the recorded solver evals
/// with the generated seed candidates (a seed-startup eval lands exactly on a
/// candidate; interior trial steps and the converged optimum do not).
#[test]
fn parsimony_multistart_breaks_after_sharp_well_penalized_first_seed() {
    fn seeds_run(center: f64) -> (usize, OuterResult) {
        let mut seed_config = gam_problem::SeedConfig::default();
        seed_config.seed_budget = 2;
        seed_config.risk_profile = gam_problem::SeedRiskProfile::GeneralizedLinear;
        let candidates: Vec<Array1<f64>> =
            crate::seeding::generate_rho_candidates(1, None, &seed_config).expect("ordered seed bounds");
        // The optimum must not coincide with any generated seed, so only true
        // seed-startup evals (which land exactly on a candidate) are counted.
        assert!(
            candidates.iter().all(|c| (c[0] - center).abs() > 1e-9),
            "test premise: the optimum {center} must not equal a generated seed"
        );
        let started = Arc::new(Mutex::new(Vec::<Array1<f64>>::new()));
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Either)
            .with_seed_config(seed_config)
            .with_max_iter(16);
        let mut obj = problem.build_objective(
            (),
            move |_: &mut (), theta: &Array1<f64>| {
                let d = theta[0] - center;
                Ok(0.5 * d * d)
            },
            {
                let started = Arc::clone(&started);
                move |_: &mut (), theta: &Array1<f64>| {
                    started.lock().unwrap().push(theta.clone());
                    let d = theta[0] - center;
                    Ok(OuterEval {
                        cost: 0.5 * d * d,
                        gradient: array![d],
                        hessian: HessianValue::Dense(array![[1.0]]),
                        inner_beta_hint: None,
                    })
                }
            },
            None::<fn(&mut ())>,
            None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let result = problem
            .run(&mut obj, "parsimony multistart wiring")
            .expect("a strictly-convex quadratic outer objective converges");
        let starts = started.lock().unwrap();
        let mut origins: Vec<f64> = starts
            .iter()
            .filter(|t| candidates.iter().any(|c| (c[0] - t[0]).abs() < 1e-9))
            .map(|t| t[0])
            .collect();
        origins.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        origins.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        (origins.len(), result)
    }

    // Well-penalized minimum (ρ = 2.7 ≥ 0): slot 0 is sharp and every λ ≥ 1, so
    // the heavy seed is redundant — exactly ONE seed solves.
    let (well_penalized_seeds, well_result) = seeds_run(2.7);
    assert!(well_result.converged(), "well-penalized fit converges");
    assert!(
        (well_result.rho[0] - 2.7).abs() < 1e-4,
        "publishes the slot-0 optimum, got {}",
        well_result.rho[0]
    );
    assert_eq!(
        well_penalized_seeds, 1,
        "a sharp, well-penalized slot-0 optimum must break the multi-start after one seed (#1575)"
    );

    // Under-penalized minimum (ρ = -2.7 < 0): the #1373 overshoot regime — the
    // heavy parsimony seed must still run.
    let (under_penalized_seeds, under_result) = seeds_run(-2.7);
    assert!(under_result.converged(), "under-penalized fit converges");
    assert!(
        (under_result.rho[0] + 2.7).abs() < 1e-4,
        "publishes the slot-0 optimum, got {}",
        under_result.rho[0]
    );
    assert_eq!(
        under_penalized_seeds, 2,
        "an under-penalized (ρ<0) slot-0 optimum must keep the parsimony second seed (#1373)"
    );
}

#[test]
fn run_starts_solver_with_direct_startup_eval() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    let calls = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let calls = Arc::clone(&calls);
            move |_: &mut (), theta: &Array1<f64>| {
                calls.lock().unwrap().push("cost");
                Ok(theta[0] * theta[0])
            }
        },
        {
            let calls = Arc::clone(&calls);
            move |_: &mut (), theta: &Array1<f64>| {
                calls.lock().unwrap().push("eval");
                Ok(OuterEval {
                    cost: theta[0] * theta[0],
                    gradient: array![2.0 * theta[0]],
                    hessian: HessianValue::Dense(array![[2.0]]),
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    // This test pins the STARTUP eval ORDER, not convergence. The single-iter
    // budget leaves a small residual gradient above the tight stationarity bound,
    // so the run may legitimately refuse to certify — that outcome is orthogonal
    // to what is asserted here. The `calls` trace records the startup sequence
    // whether or not the run mints, so the run's Result is deliberately ignored.
    drop(problem.run(&mut obj, "solver should start from a direct startup eval"));
    let calls = calls.lock().unwrap();
    let first_eval_idx = calls
        .iter()
        .position(|call| *call == "eval")
        .expect("solver should eventually request a full eval");
    assert!(
        first_eval_idx == 0,
        "startup should not perform a separate cost-screening pass first: {calls:?}"
    );
}

#[test]
fn run_screening_reorders_expensive_generated_seeds_before_full_startup_eval() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 2;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::GeneralizedLinear;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config).expect("ordered seed bounds")
        .last()
        .expect("seed generator should yield at least one candidate")
        .clone();
    let started = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let valid_seed = valid_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                if theta == valid_seed {
                    Ok(0.0)
                } else {
                    Ok(1000.0)
                }
            }
        },
        {
            let valid_seed = valid_seed.clone();
            let started = Arc::clone(&started);
            move |_: &mut (), theta: &Array1<f64>| {
                started.lock().unwrap().push(theta.clone());
                if theta == valid_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: array![0.0],
                        hessian: HessianValue::Dense(array![[1.0]]),
                        inner_beta_hint: None,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "screening should reorder expensive seeds")
        .expect("screened startup should reach the best generated seed");
    assert_eq!(result.rho, valid_seed);
    let started_snapshot: Vec<Array1<f64>> = started.lock().unwrap().clone();
    // The interior-extreme promotion (#1074/#1373/#1426) reserves slot 0 for the
    // most-flexible interior seed and slot 1 for the heaviest, so screening's
    // cost rank resumes at slot 2. (This promotion runs INSIDE
    // `rank_seeds_with_screening`, so its footprint at slots 0/1 — here the
    // generator's `[0.0]` and `[12.0]`, NOT the raw generator-first `[1.0]` — is
    // itself proof that screening ran.) The lowest-cost generated seed must lead
    // that reorderable tail: screening moved it ahead of the other equal-or-
    // higher-cost seeds it is allowed to reorder, exactly as the original "front"
    // assertion intended before the promotion reserved the first two slots.
    assert_eq!(
        started_snapshot.get(2).cloned(),
        Some(valid_seed),
        "screening should rank the lowest-cost seed at the head of the reorderable \
         tail (slots 0/1 are reserved for the promoted flexible/heaviest seeds); \
         started order was {started_snapshot:?}",
    );
    assert_eq!(screening_cap.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn thrown_screening_error_is_fatal_across_multistart_and_solver_plans() {
    const SENTINEL: &str = "fatal outer evaluation sentinel";

    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 2;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::GeneralizedLinear;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let calls = Arc::clone(&calls);
            move |_: &mut (), _: &Array1<f64>| {
                calls.fetch_add(1, Ordering::Relaxed);
                Err(EstimationError::InvalidInput(SENTINEL.to_string()))
            }
        },
        |_: &mut (), _: &Array1<f64>| -> Result<OuterEval, EstimationError> {
            panic!("a fatal screening error must prevent full outer evaluation")
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );

    let error = match problem.run(&mut obj, "fatal screening error") {
        Err(error) => error,
        Ok(_) => panic!("a fatal screening error unexpectedly minted an outer result"),
    };
    assert!(error.is_fatal_outer_evaluation(), "{error}");
    assert!(error.to_string().contains(SENTINEL), "{error}");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "a thrown evaluator error must not be replayed across seeds, cap stages, or solver plans"
    );
    assert_eq!(
        screening_cap.load(Ordering::Relaxed),
        0,
        "fatal screening exit must restore the caller's inner-iteration cap"
    );
}

#[test]
fn initial_rho_with_single_seed_budget_skips_expensive_screening() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 1;
    // This test asserts the `initial_rho + seed_budget==1` screening-skip
    // (`explicit_initial_rho_owns_single_seed_budget`) fires. That skip keys off
    // the EFFECTIVE budget, not the requested one. Pin the fixture to Gaussian,
    // whose `effective_seed_budget` is 1, so the `seed_budget == 1` skip guard is
    // true and the skip is genuinely exercised — the behaviour this test guards.
    // (A profile whose effective budget were > 1 would make the guard false and
    // let screening run instead; Arc Gaussian and GLM are both floored to 1.)
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let screening_calls = Arc::new(AtomicUsize::new(0));
    let initial_seed = array![9.0];
    let started = Arc::new(Mutex::new(Vec::new()));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_initial_rho(initial_seed.clone())
        // Declare a problem size whose estimated PSIS work trips the terminal
        // rho-uncertainty diagnostic cost gate, so its 32 `eval_cost` samples do
        // NOT run here. This test isolates the SEED-SCREENING accounting (screening
        // is skipped: `screening_cap == 0` and `screening_calls == 0`); the
        // mandatory terminal value audit and post-certification uncertainty
        // diagnostic are separate phases and must not be counted as screening.
        .with_problem_size(1_000_000, 1)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let screening_calls = Arc::clone(&screening_calls);
            let screening_cap = Arc::clone(&screening_cap);
            move |_: &mut (), _: &Array1<f64>| {
                if screening_cap.load(Ordering::Relaxed) != 0 {
                    screening_calls.fetch_add(1, Ordering::Relaxed);
                }
                Ok(0.0)
            }
        },
        {
            let started = Arc::clone(&started);
            let initial_seed = initial_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                started.lock().unwrap().push(theta.clone());
                if theta == initial_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: array![0.0],
                        hessian: HessianValue::Dense(array![[1.0]]),
                        inner_beta_hint: None,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "initial rho should be authoritative")
        .expect("initial-rho startup should not spend seed-screening solves");
    assert_eq!(result.rho, initial_seed);
    assert_eq!(
        screening_calls.load(Ordering::Relaxed),
        0,
        "explicit initial rho plus seed_budget=1 should skip screening"
    );
    assert_eq!(
        started.lock().unwrap().first().cloned(),
        Some(initial_seed),
        "solver should start from the explicit initial rho"
    );
    assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
}

#[test]
fn run_screening_reorders_bfgs_seeds_before_full_startup_eval() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let initial_seed = array![9.0];
    let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config).expect("ordered seed bounds")
        .first()
        .expect("seed generator should yield at least one candidate")
        .clone();
    let started = Arc::new(Mutex::new(Vec::new()));
    let screening_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_initial_rho(initial_seed)
        .with_screen_initial_rho(true)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let valid_seed = valid_seed.clone();
            let screening_calls = Arc::clone(&screening_calls);
            move |_: &mut (), theta: &Array1<f64>| {
                screening_calls.fetch_add(1, Ordering::Relaxed);
                if theta == valid_seed {
                    Ok(0.0)
                } else {
                    Ok(1000.0)
                }
            }
        },
        {
            let valid_seed = valid_seed.clone();
            let started = Arc::clone(&started);
            move |_: &mut (), theta: &Array1<f64>| {
                started.lock().unwrap().push(theta.clone());
                if theta == valid_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: array![0.0],
                        hessian: HessianValue::Unavailable,
                        inner_beta_hint: None,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let result = problem
        .run(&mut obj, "BFGS screening should reorder expensive seeds")
        .expect("screened BFGS startup should reach the best generated seed");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    assert_eq!(result.rho, valid_seed);
    let started_snapshot: Vec<Array1<f64>> = started.lock().unwrap().clone();
    // As in the analytic-gradient sibling test: the interior-extreme promotion
    // (#1074/#1373/#1426) reserves slot 0 (most-flexible interior seed) and slot 1
    // (heaviest interior seed — here the screened-in initial ρ=9.0), so screening's
    // cost rank resumes at slot 2. The lowest-cost generated seed must lead that
    // reorderable tail — screening moved it ahead of every other equal-or-higher-
    // cost seed it is allowed to reorder.
    assert_eq!(
        started_snapshot.get(2).cloned(),
        Some(valid_seed),
        "BFGS screening should rank the lowest-cost seed at the head of the \
         reorderable tail (slots 0/1 are reserved for the promoted flexible/heaviest \
         seeds); started order was {started_snapshot:?}",
    );
    assert!(
        screening_calls.load(Ordering::Relaxed) > 1,
        "BFGS seed screening should rank candidates with cost-only probes first",
    );
    assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
}

#[test]
fn screening_cap_survives_per_seed_reset_before_proxy_eval() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 3;
    seed_config.seed_budget = 1;
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let proxy_saw_cap = Arc::new(AtomicBool::new(false));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_max_iter(1);
    let mut obj = problem.build_objective_with_screening_proxy(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(0.0),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: theta[0].abs(),
                gradient: array![0.0],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        |_: &mut (), theta: &Array1<f64>, _: OuterEvalOrder| {
            Ok(OuterEval {
                cost: theta[0].abs(),
                gradient: array![0.0],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        {
            let screening_cap = Arc::clone(&screening_cap);
            Some(move |_: &mut ()| {
                screening_cap.store(0, Ordering::Relaxed);
            })
        },
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        {
            let screening_cap = Arc::clone(&screening_cap);
            let proxy_saw_cap = Arc::clone(&proxy_saw_cap);
            move |_: &mut (), theta: &Array1<f64>| {
                let cap = screening_cap.load(Ordering::Relaxed);
                if cap > 0 {
                    proxy_saw_cap.store(true, Ordering::Relaxed);
                    Ok(theta[0].abs())
                } else {
                    Err(EstimationError::RemlOptimizationFailed(
                        "screening proxy ran without an active cap".to_string(),
                    ))
                }
            }
        },
    );
    problem
        .run(&mut obj, "screening cap reset regression")
        .expect("screening cap should be restored after each per-seed reset");
    assert!(
        proxy_saw_cap.load(Ordering::Relaxed),
        "screening proxy should observe a nonzero cap"
    );
    assert_eq!(screening_cap.load(Ordering::Relaxed), 0);
}

#[test]
fn rank_seeds_cascade_escalates_when_initial_cap_collapses_all() {
    // When every seed's cost is non-finite at the initial screening cap
    // we must NOT jump straight to a fully uncapped re-evaluation on
    // every seed (the original two-stage protocol). Instead the cap
    // should escalate geometrically (initial → 4× → 16× → uncapped),
    // exiting the moment any cap stage produces a finite cost. This
    // test forces a cost function that returns non-finite for cap < 12
    // and finite for cap ≥ 12, then asserts the cascade exits at the
    // 4× stage with a meaningful ranking — never reaching the uncapped
    // pass.
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 1;
    seed_config.screen_max_inner_iterations = 3;
    let screening_cap = Arc::new(AtomicUsize::new(0));
    let initial_seed = array![5.0];
    let valid_seed = crate::seeding::generate_rho_candidates(1, None, &seed_config).expect("ordered seed bounds")
        .first()
        .expect("seed generator should yield at least one candidate")
        .clone();
    let max_cap_seen = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_seed_config(seed_config)
        .with_screening_cap(Arc::clone(&screening_cap))
        .with_initial_rho(initial_seed.clone())
        .with_screen_initial_rho(true)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        {
            let screening_cap = Arc::clone(&screening_cap);
            let max_cap_seen = Arc::clone(&max_cap_seen);
            let valid_seed = valid_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                let cap = screening_cap.load(Ordering::Relaxed);
                max_cap_seen.fetch_max(cap, Ordering::Relaxed);
                // Mimic an inner solver that needs ≥ 12 iterations of
                // budget to certify a finite cost; below that it returns
                // a non-finite "could not converge" signal.
                if cap > 0 && cap < 12 {
                    return Ok(f64::NAN);
                }
                if theta == valid_seed {
                    Ok(0.0)
                } else {
                    Ok(1000.0)
                }
            }
        },
        {
            let valid_seed = valid_seed.clone();
            move |_: &mut (), theta: &Array1<f64>| {
                if theta == valid_seed {
                    Ok(OuterEval {
                        cost: 0.0,
                        gradient: array![0.0],
                        hessian: HessianValue::Dense(array![[1.0]]),
                        inner_beta_hint: None,
                    })
                } else {
                    Ok(OuterEval::infeasible(theta.len()))
                }
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem
        .run(&mut obj, "cascade should escalate")
        .expect("cascade should reach a finite cost at the 4× cap stage");
    // The cascade is [3, 12, 48, 0]; the 4× stage (cap=12) is the first
    // stage that produces a finite cost, so the cascade must exit there
    // and never escalate to 48 or to the uncapped (0) stage.
    let max_cap = max_cap_seen.load(Ordering::Relaxed);
    assert_eq!(
        max_cap, 12,
        "cascade should stop at the 4× cap stage; observed max cap = {max_cap}"
    );
    assert_eq!(
        screening_cap.load(Ordering::Relaxed),
        0,
        "screening cap must be restored to its previous value after cascade"
    );
}

#[test]
fn run_typed_efs_runtime_fallback_degrades_to_bfgs_immediately() {
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.seed_budget = 2;
    let efs_calls = Arc::new(AtomicUsize::new(0));
    let problem = OuterProblem::new(12)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_initial_rho(Array1::zeros(12))
        .with_max_iter(5);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(0.5 * theta.dot(theta)),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.5 * theta.dot(theta),
                gradient: theta.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        {
            let efs_calls = Arc::clone(&efs_calls);
            Some(move |_: &mut (), _: &Array1<f64>| {
                efs_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // The EFS bridge translates typed gradient unavailability into
                // a typed first-order fallback request; no message token
                // participates in routing.
                Err(EstimationError::GradientUnavailable {
                    context: "synthetic EFS runtime escape hatch",
                    mode: "efs runtime escape hatch",
                })
            })
        },
    );
    let result = problem
        .run(&mut obj, "EFS runtime fallback request")
        .expect("runtime EFS escape hatch should degrade to BFGS");
    assert_eq!(result.plan_used.solver, Solver::Bfgs);
    assert_eq!(
        efs_calls.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "runtime fallback request should abort the EFS attempt immediately"
    );
}

#[test]
fn run_rejects_invalid_theta_layout() {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_psi_dim(2)
        .with_initial_rho(Array1::zeros(1))
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(0.0),
        |_: &mut (), _: &Array1<f64>| {
            Ok(OuterEval {
                cost: 0.0,
                gradient: Array1::zeros(1),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let err = problem
        .run(&mut obj, "test invalid layout")
        .expect_err("invalid theta layout should fail cleanly");
    assert!(
        err.to_string().contains("invalid outer theta layout"),
        "unexpected error: {err}"
    );
}

#[test]
fn effective_seed_budget_caps_expensive_solver_retries() {
    assert_eq!(
        effective_seed_budget(
            4,
            Solver::Efs,
            gam_problem::SeedRiskProfile::GeneralizedLinear,
        ),
        1
    );
    assert_eq!(
        effective_seed_budget(4, Solver::HybridEfs, gam_problem::SeedRiskProfile::Survival,),
        1
    );
    // #2376: Arc + a parsimonious profile (GeneralizedLinear / Survival) keeps
    // the REQUESTED budget, so the #1373/#1575 promoted heavy interior seed at
    // slot 1 stays reachable. Flooring these to 1 (the former #1575/#1074/#1426
    // "the initial.sp seed reaches the heavily-penalized GLM basin" assumption)
    // made the multi-start await gate's `seed_budget > 1` unsatisfiable and
    // silently disabled the under-penalized-overshoot guard. The single-seed
    // speed win for the common well-penalized case is now reclaimed at RUNTIME
    // by `parsimony_second_seed_is_redundant`, not by capping the budget here.
    assert_eq!(
        effective_seed_budget(
            3,
            Solver::Arc,
            gam_problem::SeedRiskProfile::GeneralizedLinear,
        ),
        3
    );
    assert_eq!(
        effective_seed_budget(3, Solver::Arc, gam_problem::SeedRiskProfile::Survival,),
        3
    );
    // A caller that genuinely requests a single start still gets one: the
    // parsimony second seed is only re-enabled when a budget ≥ 2 was asked for.
    assert_eq!(
        effective_seed_budget(
            1,
            Solver::Arc,
            gam_problem::SeedRiskProfile::GeneralizedLinear,
        ),
        1
    );
    // #2623: exact curvature proves only local convergence. Gaussian therefore
    // retains the requested analytic-candidate budget so distinct basins are
    // compared by their converged REML values.
    assert_eq!(
        effective_seed_budget(3, Solver::Arc, gam_problem::SeedRiskProfile::Gaussian),
        3
    );
    // GaussianLocationScale is NOT floored (it uses lowest-cost keep-best but
    // its promoted-seed multi-start needs budget ≥ 2); it falls through to the
    // requested budget, matching the behaviour before #2376.
    assert_eq!(
        effective_seed_budget(
            3,
            Solver::Arc,
            gam_problem::SeedRiskProfile::GaussianLocationScale,
        ),
        3
    );
    assert_eq!(
        effective_seed_budget(3, Solver::Bfgs, gam_problem::SeedRiskProfile::Survival,),
        3
    );

    let bfgs_glm = gam_problem::SeedConfig {
        seed_budget: 3,
        risk_profile: gam_problem::SeedRiskProfile::GeneralizedLinear,
        num_auxiliary_trailing: 0,
        ..Default::default()
    };
    assert_eq!(
        effective_seed_budget_for_config(&bfgs_glm, Solver::Bfgs),
        1,
        "gradient-only GLM BFGS owns one neutral start (#2519)",
    );

    let bfgs_glm_with_aux = gam_problem::SeedConfig {
        num_auxiliary_trailing: 1,
        ..bfgs_glm
    };
    assert_eq!(
        effective_seed_budget_for_config(&bfgs_glm_with_aux, Solver::Bfgs),
        3,
        "a trailing auxiliary coordinate preserves the caller's multistart policy",
    );
}

#[test]
fn bfgs_glm_single_start_prioritizes_the_neutral_seed_2519() {
    let config = gam_problem::SeedConfig {
        seed_budget: 2,
        risk_profile: gam_problem::SeedRiskProfile::GeneralizedLinear,
        num_auxiliary_trailing: 0,
        ..Default::default()
    };
    let mut seeds = vec![array![-2.0, -2.0, -2.0], array![0.0, 0.0, 0.0]];
    prioritize_neutral_bfgs_glm_seed(&mut seeds, &config, Solver::Bfgs, 1);
    assert_eq!(seeds[0], array![0.0, 0.0, 0.0]);

    let unchanged = seeds.clone();
    prioritize_neutral_bfgs_glm_seed(&mut seeds, &config, Solver::Arc, 1);
    assert_eq!(
        seeds, unchanged,
        "ARC retains its flexible/heavy basin policy",
    );
}

#[test]
fn rejected_budgeted_starts_license_bounded_replacement_seeds_2519() {
    assert!(should_start_next_seed(0, 1, false));
    assert!(
        should_start_next_seed(1, 1, false),
        "a rejected nominal start must not end the search without a certified candidate",
    );
    assert!(
        !should_start_next_seed(1, 1, true),
        "the first certified candidate exhausts the single-start policy",
    );
    assert!(
        should_start_next_seed(2, 1, false),
        "the finite generated seed list, not a failed-start count, bounds replacements",
    );
}

#[test]
fn run_arc_projects_seed_before_seed_validation_eval() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 1;
    seed_config.seed_budget = 1;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Either)
        .with_bounds(array![0.0], array![1.0])
        .with_initial_rho(array![2.0])
        .with_seed_config(seed_config)
        // The subject is WHICH point the first evaluation sees, and that is
        // settled before the optimizer takes any step — `seen.first()` proves
        // the ordering on its own. The iteration budget is incidental here, and
        // at `1` it was actively harmful: one ARC step from the projected seed
        // `1.0` toward the optimum `0.25` lands at `0.2504` with
        // `|Pg| = 8.541e-4` against a `6.325e-4` stationarity bound — 1.35×
        // short — so the run refused ("claimed_converged=false after 1 outer
        // iteration(s)") and the `expect` below fired on a convergence budget
        // that has nothing to do with seed projection. Give the quadratic room
        // to certify, so projection is the only thing that can fail here.
        .with_max_iter(16);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok((theta[0] - 0.25).powi(2)),
        {
            let seen = Arc::clone(&seen);
            move |_: &mut (), theta: &Array1<f64>| {
                seen.lock().unwrap().push(theta.clone());
                Ok(OuterEval {
                    cost: (theta[0] - 0.25).powi(2),
                    gradient: array![2.0 * (theta[0] - 0.25)],
                    hessian: HessianValue::Dense(array![[2.0]]),
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    problem
        .run(&mut obj, "arc seed projection")
        .expect("arc should evaluate the projected seed");
    assert_eq!(
        seen.lock().unwrap().first().cloned(),
        Some(array![1.0]),
        "Arc must project the seed before validating the initial sample",
    );
}

#[test]
fn run_bfgs_projects_seed_before_seed_validation_eval() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 1;
    seed_config.seed_budget = 1;
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![0.0], array![1.0])
        .with_initial_rho(array![2.0])
        .with_seed_config(seed_config)
        .with_max_iter(1);
    let mut obj = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok((theta[0] - 0.25).powi(2)),
        {
            let seen = Arc::clone(&seen);
            move |_: &mut (), theta: &Array1<f64>| {
                seen.lock().unwrap().push(theta.clone());
                Ok(OuterEval {
                    cost: (theta[0] - 0.25).powi(2),
                    gradient: array![2.0 * (theta[0] - 0.25)],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    // This test pins seed PROJECTION (the initial ρ=[2.0] is clamped to the box
    // upper bound [1.0] before the first sample eval), not convergence. The
    // single-iter budget from the projected seed need not reach the [0.25]
    // optimum, so the run may legitimately refuse to certify — orthogonal to the
    // projection assertion. The `seen` trace records the first evaluated point
    // whether or not the run mints, so the run's Result is deliberately ignored.
    drop(problem.run(&mut obj, "bfgs seed projection"));
    assert_eq!(
        seen.lock().unwrap().first().cloned(),
        Some(array![1.0]),
        "BFGS must project the seed before validating the initial sample",
    );
}

/// #2569 — a seed the certify-resume loop has already started and already had
/// refused must be REPLAYED from the record, never re-run, and the caller's own
/// reseed point must survive that filter.
///
/// The production defect this gates: `should_start_next_seed` lets the cascade
/// continue past `seed_budget` while nothing has certified, so every resume
/// round re-entered the same regenerated lattice seed and re-derived the same
/// refusal — measured at 18-48% of the wall clock on the #2569 grouped-binomial
/// design.
#[test]
fn a_replayed_seed_refusal_never_drops_the_reseed_point_2569() {
    let reseed = array![-2.0];
    let recorded = array![3.140625];
    let untried = array![7.0];

    let (kept, replayed) = seeds_without_recorded_refusals(
        vec![reseed.clone(), recorded.clone(), untried.clone()],
        std::slice::from_ref(&recorded),
    );
    assert_eq!(replayed, 1, "the recorded seed must be replayed, not re-run");
    assert_eq!(
        kept,
        vec![reseed.clone(), untried.clone()],
        "a seed with no record must remain reachable, so the budget fall-through \
         keeps every rescue it could previously perform"
    );

    // Slot 0 is the resume's reseed point: recorded or not, it is what the
    // retry exists to explore and is never dropped.
    let (kept_slot_zero, replayed_slot_zero) = seeds_without_recorded_refusals(
        vec![reseed.clone(), untried.clone()],
        std::slice::from_ref(&reseed),
    );
    assert_eq!(replayed_slot_zero, 0);
    assert_eq!(kept_slot_zero, vec![reseed.clone(), untried.clone()]);

    // Nothing recorded: the cascade is returned unchanged, which is every
    // non-resume path in production.
    let (kept_identity, replayed_identity) =
        seeds_without_recorded_refusals(vec![reseed.clone(), recorded.clone()], &[]);
    assert_eq!(replayed_identity, 0);
    assert_eq!(kept_identity, vec![reseed, recorded]);
}

/// #2569 — only a `"certificate"` refusal states where a seed's search
/// TERMINATED. A seed rejected at screening, domain entry or validation never
/// reached a solver, so re-running it later is not redundant work and it must
/// not be suppressed.
#[test]
fn only_certificate_phase_seed_refusals_are_recorded_2569() {
    let error = EstimationError::RemlOptimizationFailed("refused".to_string());
    let seeds = vec![array![-2.0], array![3.140625], array![7.0]];
    let rejections = vec![
        SeedRejection::from_estimation_error(1, "certificate", &error),
        // A repeat of the same seed collapses: the list is a set of points.
        SeedRejection::from_estimation_error(1, "certificate", &error),
        // Never reached a solver — says nothing about where it would stop.
        SeedRejection::from_estimation_error(2, "validation", &error),
        // Out of range (a rejection recorded against a seed list this call does
        // not own) is dropped rather than panicking.
        SeedRejection::from_estimation_error(9, "certificate", &error),
    ];

    assert_eq!(
        certificate_refused_seed_points(&rejections, &seeds),
        vec![array![3.140625]],
    );
    assert!(certificate_refused_seed_points(&[], &seeds).is_empty());
}

/// #2569 end-to-end, with the positive control the claim needs: the cascade
/// DOES fall through past its budget of 1 when nothing has certified (control
/// arm), and does NOT re-run the fall-through seed once its refusal is on
/// record (replay arm) — while still running the caller's reseed point in both.
#[test]
fn the_cascade_replays_a_recorded_seed_instead_of_falling_through_to_it_2569() {
    // A minimum far outside the box: neither seed is stationary, so no seed can
    // certify and `best` stays `None` — the state in which the budget is inert.
    const MINIMUM: f64 = 40.0;
    let reseed = array![-2.0];
    let fall_through = array![3.140625];

    fn run_arm(
        reseed: &Array1<f64>,
        fall_through: &Array1<f64>,
        previously_refused: Vec<Array1<f64>>,
    ) -> Vec<Array1<f64>> {
        #[derive(Default)]
        struct State {
            seen: Vec<Array1<f64>>,
        }
        let problem = OuterProblem::new(1)
            .with_gradient(Derivative::Analytic)
            .with_hessian(DeclaredHessianForm::Unavailable)
            .with_initial_rho(reseed.clone())
            .with_initial_rho_candidates(vec![fall_through.clone()])
            .with_bounds(array![-8.0], array![8.0])
            .with_max_iter(1)
            .with_seed_config(gam_problem::SeedConfig {
                max_seeds: 1,
                seed_budget: 1,
                ..Default::default()
            });
        let mut obj = problem.build_objective(
            State::default(),
            move |state: &mut State, theta: &Array1<f64>| {
                state.seen.push(theta.clone());
                let delta = theta[0] - 40.0;
                Ok(0.5 * delta * delta)
            },
            move |state: &mut State, theta: &Array1<f64>| {
                state.seen.push(theta.clone());
                let delta = theta[0] - 40.0;
                Ok(OuterEval {
                    cost: 0.5 * delta * delta,
                    gradient: array![delta],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            },
            None::<fn(&mut State)>,
            None::<fn(&mut State, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
        );
        let mut config = problem.config();
        config.previously_refused_seed_points = previously_refused;
        let cap = obj.capability();
        let the_plan = plan(&cap);
        let outcome = run_outer_with_plan(
            &mut obj,
            &config,
            "2569 recorded seed replay",
            &cap,
            &the_plan,
            true,
        );
        // The minimum is outside the box, so nothing here can certify. That is
        // the state in which `should_start_next_seed` ignores the budget, and
        // it is the precondition both arms are measured under.
        assert!(
            !matches!(outcome, Ok(PlanRunOutcome::Converged(_))),
            "fixture precondition: no seed may certify, or the cascade never \
             reaches the budget fall-through this test is about"
        );
        obj.state.seen.clone()
    }

    let visited = |seen: &[Array1<f64>], point: &Array1<f64>| {
        seen.iter()
            .any(|theta| theta.len() == 1 && theta[0].to_bits() == point[0].to_bits())
    };

    let control = run_arm(&reseed, &fall_through, Vec::new());
    assert!(
        visited(&control, &reseed),
        "control: the caller's own seed must be evaluated"
    );
    assert!(
        visited(&control, &fall_through),
        "positive control: with nothing on record the cascade falls through its \
         budget of 1 onto the second seed — the very behaviour the replay must \
         suppress. If this arm fails, the fixture cannot separate the arms and \
         the assertion below proves nothing."
    );
    // The minimum is outside the box on purpose; assert it, so a fixture that
    // silently starts certifying (and therefore never falls through) is caught
    // here rather than passing the replay arm for the wrong reason.
    assert!(MINIMUM > 8.0);

    let replayed = run_arm(&reseed, &fall_through, vec![fall_through.clone()]);
    assert!(
        visited(&replayed, &reseed),
        "the resume's reseed point must still run"
    );
    assert!(
        !visited(&replayed, &fall_through),
        "a seed whose certificate refusal is already on record must not be \
         re-run: it terminates where it terminated before"
    );
}
