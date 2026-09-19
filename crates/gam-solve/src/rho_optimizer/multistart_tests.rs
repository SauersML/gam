// Child module of `run::multistart` (see the `#[path]` declaration there).
#![cfg(test)]

use super::*;
use ndarray::array;

/// A one-coordinate criterion on `[−8, 8]` with a railed basin and an interior
/// well. `V = 1 + ½(1 + s·tanh(ρ + 2s)) − 2·exp(−(ρ − c)²/2)` falls towards the
/// rail on the side `−s` points to, where it certifies at `V ≈ 1` with its
/// gradient pointing outward, and the well at `c` is lower (`V ≈ 0`).
fn railed_and_well(rho: f64, side: f64, centre: f64) -> OuterEval {
    let well = (-(rho - centre).powi(2) / 2.0).exp();
    let sech = 1.0 / (rho + side * 2.0).cosh();
    OuterEval {
        cost: 1.0 + 0.5 * (1.0 + side * (rho + side * 2.0).tanh()) - 2.0 * well,
        gradient: array![0.5 * side * sech * sech + 2.0 * (rho - centre) * well],
        hessian: HessianValue::Unavailable,
        inner_beta_hint: None,
    }
}

fn problem_with_candidates(candidates: Vec<Array1<f64>>) -> OuterProblem {
    OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho_candidates(candidates)
        .with_bounds(array![-8.0], array![8.0])
        .with_seed_config(gam_problem::SeedConfig {
            max_seeds: 1,
            seed_budget: 1,
            ..Default::default()
        })
}

/// Run one seed of a multistart on the fixture, recording every ρ it evaluated.
fn run_fixture_seed(
    problem: &OuterProblem,
    side: f64,
    centre: f64,
    delay: std::time::Duration,
) -> (Result<CertifiedOuterResult, EstimationError>, Vec<f64>) {
    let mut obj = problem.build_objective(
        Vec::<f64>::new(),
        move |_: &mut Vec<f64>, theta: &Array1<f64>| Ok(railed_and_well(theta[0], side, centre).cost),
        move |seen: &mut Vec<f64>, theta: &Array1<f64>| {
            seen.push(theta[0]);
            Ok(railed_and_well(theta[0], side, centre))
        },
        None::<fn(&mut Vec<f64>)>,
        None::<fn(&mut Vec<f64>, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    std::thread::sleep(delay);
    let outcome = problem.run_certified(&mut obj, "multistart fixture");
    (outcome, obj.state.clone())
}

/// gnomon#2359 stiff-basin control. The neutral seed ρ = 0 and the GLM anchor
/// ρ = 1 fall to the lower rail and certify at `V ≈ 1`; the lower well at ρ = 6
/// drains only the stiff ρ = 4 seed, which starts above the railed value. The
/// cascade's own rules searched the neutral seed alone and published the rail
/// (`a_railed_winner_does_not_hide_a_basin_only_a_stiff_seed_reaches_2359`
/// measured `rho=-8 value=1.0000061` under the screened-rank rule). Searching
/// every seed publishes the well.
#[test]
fn multistart_publishes_the_basin_only_a_stiff_seed_reaches_2359() {
    let problem = problem_with_candidates(vec![array![0.0], array![4.0]]);
    let seeds = problem.outer_seeds("stiff control").expect("seeds");
    assert!(
        seeds.iter().any(|seed| seed[0] == 0.0) && seeds.iter().any(|seed| seed[0] == 4.0),
        "fixture precondition: both the neutral and the stiff seed are generated, got {seeds:?}"
    );
    let outcome = problem
        .run_certified_multistart("stiff control", 0, 1, |_, seed_problem, _| {
            run_fixture_seed(&seed_problem, 1.0, 6.0, std::time::Duration::ZERO)
        })
        .expect("multistart runs");
    let winner = outcome.winner.expect("a seed certifies");
    let published = outcome.runs[winner].0.as_ref().expect("the winner certified");
    assert!(
        (published.rho()[0] - 6.0).abs() < 0.1 && published.final_value() < 1.0e-3,
        "published rho={} value={}: the well at rho = 6 must win",
        published.rho()[0],
        published.final_value(),
    );
    assert_eq!(outcome.seeds[winner][0], 4.0, "the stiff seed reaches the well");
    for (index, (run, searched)) in outcome.runs.iter().enumerate() {
        assert!(run.is_ok(), "seed {index} certifies: {:?}", run.as_ref().err());
        assert!(!searched.is_empty(), "seed {index} ran its own search");
    }
}

/// The #2359 shape, mirrored: the neutral seed falls to the upper rail and the
/// flexible ρ = −2 seed, which starts above that railed value, reaches the well
/// at ρ = −5. Searching every seed publishes the well.
#[test]
fn multistart_publishes_the_basin_only_a_flexible_seed_reaches_2359() {
    let problem = problem_with_candidates(vec![array![0.0], array![-2.0]]);
    let outcome = problem
        .run_certified_multistart("flexible control", 0, 1, |_, seed_problem, _| {
            run_fixture_seed(&seed_problem, -1.0, -5.0, std::time::Duration::ZERO)
        })
        .expect("multistart runs");
    let winner = outcome.winner.expect("a seed certifies");
    let published = outcome.runs[winner].0.as_ref().expect("the winner certified");
    assert!(
        (published.rho()[0] + 5.0).abs() < 0.1 && published.final_value() < 1.0e-3,
        "published rho={} value={}: the well at rho = -5 must win",
        published.rho()[0],
        published.final_value(),
    );
}

/// A sole-seed problem searches its seed and nothing else, and each multistart
/// run receives exactly one seed.
#[test]
fn a_sole_seed_problem_searches_one_seed_2359() {
    let problem = problem_with_candidates(vec![array![0.0], array![4.0]]);
    let sole = problem.clone().with_sole_seed(array![4.0]);
    assert_eq!(
        sole.outer_seeds("sole").expect("seeds"),
        vec![array![4.0]],
        "a sole-seed run has no generated seed and no candidate"
    );
    let every = problem.outer_seeds("every").expect("seeds");
    let outcome = problem
        .run_certified_multistart("sole seeds", 0, 1, |_, seed_problem, _| {
            let seeds = seed_problem.outer_seeds("sole run").expect("seeds");
            (
                Err(EstimationError::RemlOptimizationFailed("not run".to_string())),
                seeds,
            )
        })
        .expect("multistart runs");
    assert_eq!(outcome.seeds, every);
    for (index, (_, seeds)) in outcome.runs.iter().enumerate() {
        assert_eq!(seeds, &vec![every[index].clone()], "run {index} searches its own seed alone");
    }
    assert_eq!(outcome.winner, None, "no run certified, so there is no winner");
}


/// The winner is a function of the seed runs' results alone, not of the order
/// they finish in: with the finishing order of the seed runs reversed, the same
/// seed wins with the same bits, and it is the seed keep-best picks by value.
#[test]
fn a_multistart_winner_does_not_depend_on_which_run_finishes_first_2359() {
    let problem = problem_with_candidates(vec![array![0.0], array![3.5], array![4.0]]);
    let seeds = problem.outer_seeds("finishing order").expect("seeds");
    let mut published = Vec::new();
    for reverse in [false, true] {
        let outcome = problem
            .run_certified_multistart("finishing order", 0, 1, |index, seed_problem, _| {
                let rank = if reverse { seeds.len() - index } else { index };
                run_fixture_seed(
                    &seed_problem,
                    1.0,
                    6.0,
                    std::time::Duration::from_millis(40 * rank as u64),
                )
            })
            .expect("multistart runs");
        let winner = outcome.winner.expect("a seed certifies");
        let result = outcome.runs[winner].0.as_ref().expect("the winner certified");
        let lowest = outcome
            .runs
            .iter()
            .filter_map(|(run, _)| run.as_ref().ok().map(CertifiedOuterResult::final_value))
            .fold(f64::INFINITY, f64::min);
        assert!(
            result.final_value() - lowest
                <= outer_value_agreement_bound(result.final_value(), lowest),
            "keep-best publishes the lowest certified value, to its rounding envelope"
        );
        published.push((winner, result.rho()[0].to_bits(), result.final_value().to_bits()));
    }
    assert_eq!(
        published[0], published[1],
        "the same seed wins with the same bits whichever run finished first"
    );
}

/// `V = max(|ρ − 6| − ½, 0)²`: a well whose floor is exactly zero on
/// `[5.5, 6.5]`, so seeds on the floor certify at bit-identical values.
fn flat_well(rho: f64) -> OuterEval {
    let excess = ((rho - 6.0).abs() - 0.5).max(0.0);
    OuterEval {
        cost: excess * excess,
        gradient: array![2.0 * excess * (rho - 6.0).signum()],
        hessian: HessianValue::Unavailable,
        inner_beta_hint: None,
    }
}

fn run_flat_well_seed(
    problem: &OuterProblem,
    delay: std::time::Duration,
) -> (Result<CertifiedOuterResult, EstimationError>, usize) {
    let mut obj = problem.build_objective(
        0usize,
        |_: &mut usize, theta: &Array1<f64>| Ok(flat_well(theta[0]).cost),
        |evaluations: &mut usize, theta: &Array1<f64>| {
            *evaluations += 1;
            Ok(flat_well(theta[0]))
        },
        None::<fn(&mut usize)>,
        None::<fn(&mut usize, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    std::thread::sleep(delay);
    let outcome = problem.run_certified(&mut obj, "flat well");
    (outcome, obj.state)
}

/// A tie goes to the lower seed index whatever order the runs finish in: the
/// seeds on the well's floor certify at exactly zero, and the first of them in
/// seed order wins in both finishing orders.
#[test]
fn a_multistart_tie_goes_to_the_lower_seed_index_2359() {
    let problem = problem_with_candidates(vec![array![5.75], array![6.25]]);
    let seeds = problem.outer_seeds("tie").expect("seeds");
    let first_on_floor = seeds
        .iter()
        .position(|seed| (seed[0] - 6.0).abs() < 0.5)
        .expect("fixture precondition: a seed sits on the well's floor");
    for reverse in [false, true] {
        let outcome = problem
            .run_certified_multistart("tie", 0, 1, |index, seed_problem, _| {
                let rank = if reverse { seeds.len() - index } else { index };
                run_flat_well_seed(&seed_problem, std::time::Duration::from_millis(40 * rank as u64))
            })
            .expect("multistart runs");
        let on_floor: Vec<usize> = outcome
            .runs
            .iter()
            .enumerate()
            .filter(|(_, (run, _))| run.as_ref().is_ok_and(|certified| certified.final_value() == 0.0))
            .map(|(index, _)| index)
            .collect();
        assert!(on_floor.len() >= 2, "fixture precondition: at least two exact ties, got {on_floor:?}");
        assert_eq!(
            outcome.winner,
            Some(first_on_floor),
            "the lowest-index tied seed wins (reverse finishing order: {reverse})"
        );
    }
}

/// Only certified runs compete: a run that did not certify never wins, however
/// its outcome reads, and when no seed certifies the multistart refuses with
/// every seed's outcome instead of publishing the best uncertified run.
#[test]
fn only_certified_runs_compete_and_none_certified_is_a_refusal_2359() {
    let problem = problem_with_candidates(vec![array![0.0], array![4.0]]);
    let seeds = problem.outer_seeds("certified only").expect("seeds");
    let stiff = seeds
        .iter()
        .position(|seed| seed[0] == 4.0)
        .expect("fixture precondition: the stiff seed is generated");
    // The stiff seed, the only one that reaches the well, does not certify here.
    let outcome = problem
        .run_certified_multistart("certified only", 0, 1, |index, seed_problem, _| {
            if index == stiff {
                (
                    Err(EstimationError::RemlOptimizationFailed(
                        "planted: this seed's search did not certify".to_string(),
                    )),
                    Vec::new(),
                )
            } else {
                run_fixture_seed(&seed_problem, 1.0, 6.0, std::time::Duration::ZERO)
            }
        })
        .expect("multistart runs");
    let winner = outcome.winner.expect("the other seeds certify on the rail");
    assert_ne!(winner, stiff, "an uncertified run never wins");
    let published = outcome.runs[winner].0.as_ref().expect("the winner certified");
    assert!(published.final_value() > 0.99, "the certified rail wins: {}", published.final_value());

    let refused = problem
        .run_certified_multistart("none certified", 0, 1, |index, _, _| {
            (
                Err(EstimationError::RemlOptimizationFailed(format!(
                    "planted: seed {index} did not certify"
                ))),
                (),
            )
        })
        .expect("multistart runs");
    assert_eq!(refused.winner, None);
    let message = refused.refusal("none certified").to_string();
    let mut previous = 0;
    for index in 0..seeds.len() {
        let entry = format!("seed {index} rho={:?}: not certified [", seeds[index].to_vec());
        let at = message
            .find(&entry)
            .unwrap_or_else(|| panic!("the refusal carries seed {index}'s start and verdict: {message}"));
        assert!(at >= previous, "the refusal lists seeds in seed order: {message}");
        previous = at;
        assert!(
            message.contains(&format!("planted: seed {index} did not certify")),
            "the refusal carries seed {index}'s outcome: {message}"
        );
    }
    assert_eq!(
        message.matches("[EstimationError::RemlOptimizationFailed]").count(),
        seeds.len(),
        "every seed's verdict is named by its variant: {message}"
    );
}

/// The availability the admission tests read before launch.
const PRE_LAUNCH_AVAILABLE: u64 = 123_456_789;

/// Run the three-seed fixture multistart against `governor` on a pool of
/// `width`, each search holding its lane for `hold` so that searches the budget
/// admits together overlap. Returns the outcome's winner bits, how many searches
/// were live at once, and every search's lane `(serial available, granted)`.
fn admitted_fixture_run(
    governor: &gam_runtime::resource::MemoryGovernor,
    working_set_bytes: usize,
    width: usize,
    hold: std::time::Duration,
) -> ((usize, u64, u64), usize, Vec<(u64, usize)>) {
    let problem = problem_with_candidates(vec![array![0.0], array![3.5], array![4.0]]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .build()
        .expect("a test pool");
    let outcome = pool.install(|| {
        problem
            .run_certified_multistart_on(governor, "admission", PRE_LAUNCH_AVAILABLE, working_set_bytes, |_, seed_problem, lane| {
                let (outcome, _) = run_fixture_seed(&seed_problem, 1.0, 6.0, hold);
                (outcome, (lane.serial_available_bytes(), lane.granted_bytes()))
            })
            .expect("multistart runs")
    });
    let winner = outcome.winner.expect("a seed certifies");
    let result = outcome.runs[winner].0.as_ref().expect("the winner certified");
    (
        (winner, result.rho()[0].to_bits(), result.final_value().to_bits()),
        outcome.most_live,
        outcome.runs.iter().map(|(_, lane)| *lane).collect(),
    )
}

/// Memory pressure: a budget that admits one search's working set but not two
/// makes the searches queue, one live at a time, instead of taking a smaller
/// path. Each search still gets its full working set and the same pre-launch
/// availability, and the winner is bitwise the one a budget admitting all three
/// at once publishes.
#[test]
fn a_budget_for_one_search_queues_the_rest_and_keeps_the_winner_2359() {
    let working_set = 1_000usize;
    let hold = std::time::Duration::from_millis(60);
    let tight = gam_runtime::resource::MemoryGovernor::with_budget_bytes(working_set + working_set / 2);
    let (queued_winner, queued_live, queued_lanes) = admitted_fixture_run(&tight, working_set, 3, hold);
    assert_eq!(queued_live, 1, "a budget for one search runs one at a time");
    assert!(
        queued_lanes.iter().all(|&(_, granted)| granted == working_set),
        "every queued search is granted its full working set: {queued_lanes:?}"
    );
    let ample = gam_runtime::resource::MemoryGovernor::with_budget_bytes(working_set * 16);
    let (parallel_winner, parallel_live, parallel_lanes) = admitted_fixture_run(&ample, working_set, 3, hold);
    assert!(parallel_live > 1, "an ample budget runs searches at once (most live {parallel_live})");
    assert_eq!(queued_winner, parallel_winner, "queueing does not move the winner or its bits");
    assert!(
        queued_lanes
            .iter()
            .chain(&parallel_lanes)
            .all(|&(available, _)| available == PRE_LAUNCH_AVAILABLE),
        "every search reads the availability read before launch, not a live one"
    );
}

/// A working set no budget can admit still runs, one search at a time: the
/// search with none live beside it is the serial search.
#[test]
fn a_working_set_over_the_budget_runs_the_searches_serially_2359() {
    let tiny = gam_runtime::resource::MemoryGovernor::with_budget_bytes(10);
    let (serial_winner, most_live, lanes) =
        admitted_fixture_run(&tiny, 1_000, 3, std::time::Duration::from_millis(20));
    assert_eq!(most_live, 1);
    assert!(lanes.iter().all(|&(_, granted)| granted == 0), "no grant: {lanes:?}");
    let ample = gam_runtime::resource::MemoryGovernor::with_budget_bytes(1_000_000);
    let (winner, _, _) = admitted_fixture_run(&ample, 1_000, 3, std::time::Duration::ZERO);
    assert_eq!(serial_winner, winner);
}

/// Inside a Rayon pool the lanes are that pool's tasks (the path a caller such as
/// gnomon's calibrate pool takes), outside one they are OS threads; at pool
/// widths 1, 4 and 12 and on both paths the same seed wins with the same bits,
/// and the pool's width bounds the lanes.
#[test]
fn the_winner_is_the_same_on_every_pool_width_and_lane_kind_2359() {
    let problem = problem_with_candidates(vec![array![0.0], array![3.5], array![4.0]]);
    let run = |problem: &OuterProblem| {
        let outcome = problem
            .run_certified_multistart("pool width", 0, 1, |_, seed_problem, _| {
                run_fixture_seed(&seed_problem, 1.0, 6.0, std::time::Duration::ZERO)
            })
            .expect("multistart runs");
        let winner = outcome.winner.expect("a seed certifies");
        let result = outcome.runs[winner].0.as_ref().expect("the winner certified");
        (
            outcome.lanes,
            (winner, result.rho()[0].to_bits(), result.final_value().to_bits()),
        )
    };
    let (_, outside) = run(&problem);
    for width in [1usize, 4, 12] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(width)
            .build()
            .expect("a test pool");
        let (lanes, inside) = pool.install(|| run(&problem));
        assert!(lanes <= width, "{lanes} lanes on a pool of {width}");
        assert_eq!(inside, outside, "pool width {width}: same winner and bits as OS-thread lanes");
    }
}

fn warm_start(theta: f64, value: f64, same_inputs: bool) -> gam_model_api::WarmStart {
    gam_model_api::WarmStart {
        theta: array![theta],
        beta: array![0.0],
        value,
        same_inputs,
        outcome: Default::default(),
    }
}

/// The published (seeds, ρ, V, iterations) of a multistart on the two-basin fixture.
fn published_two_basin(problem: &OuterProblem) -> (Vec<Array1<f64>>, f64, f64, usize) {
    let outcome = problem
        .run_certified_multistart("two basins", 0, 1, |_, seed_problem, _| {
            run_fixture_seed(&seed_problem, 1.0, 6.0, std::time::Duration::ZERO)
        })
        .expect("multistart runs");
    let winner = outcome.winner.expect("a seed certifies");
    let result = outcome.runs[winner]
        .0
        .as_ref()
        .expect("the winner certified");
    (
        outcome.seeds.clone(),
        result.rho()[0],
        result.final_value(),
        result.iterations(),
    )
}

/// The certified well of the two-basin fixture, searched from ρ = 4 alone: a
/// parent fit's published point and value.
fn two_basin_parent() -> CertifiedOuterResult {
    let problem =
        problem_with_candidates(vec![array![0.0], array![4.0]]).with_sole_seed(array![4.0]);
    run_fixture_seed(&problem, 1.0, 6.0, std::time::Duration::ZERO)
        .0
        .expect("the parent certifies")
}

/// A warm start from another fit's inputs joins the multistart as one more seed
/// and never replaces its seeds (gam#3002). The parent's point lies in this
/// fit's railed basin (V ≈ 1); the cold seeds reach the well (V ≈ 0). The
/// argmin over the superset publishes the well: V_warm ≤ V_cold within the
/// multistart's tie envelope, which is all a joined seed can cost. Taken as
/// the sole seed, the point publishes the rail, and this test is what catches
/// that mutant.
#[test]
fn a_warm_start_from_other_inputs_joins_the_multistart_and_never_raises_v_3002() {
    let problem = problem_with_candidates(vec![array![0.0], array![4.0]]);
    let (_, _, cold, _) = published_two_basin(&problem);
    let (_, alone_rho, alone, _) =
        published_two_basin(&problem.clone().with_sole_seed(array![-6.0]));
    assert!(
        alone > cold + 0.5,
        "fixture precondition: the parent's point alone publishes the rail \
         (rho={alone_rho} V={alone}), the cold seeds the well (V={cold})"
    );
    let parent = warm_start(-6.0, alone, false);
    let (seeds, warm_rho, warm, _) = published_two_basin(&problem.clone().with_warm_start(&parent));
    assert!(
        seeds.contains(&array![-6.0]) && seeds.contains(&array![4.0]),
        "the parent's point joins the cold seeds, got {seeds:?}"
    );
    assert!(
        warm <= cold + outer_value_agreement_bound(cold, warm),
        "the warm start raised the published V: rho={warm_rho} V={warm} against cold V={cold}"
    );
    assert_eq!(
        parent.recorded(),
        Some(gam_model_api::WarmStartOutcome::JoinedMultistart)
    );
}

/// On the parent's own inputs the parent's certified point, offered to the
/// search that certified it, is accepted where it stands: no outer iteration,
/// the parent's bits. Through the cascade and through the multistart alike.
#[test]
fn a_warm_start_on_the_parents_inputs_resumes_with_no_outer_iteration_3002() {
    let parent = two_basin_parent();
    let resume = warm_start(parent.rho()[0], parent.final_value(), true);
    let problem = problem_with_candidates(vec![array![0.0], array![4.0]]).with_warm_start(&resume);
    let (cascade, _) = run_fixture_seed(&problem, 1.0, 6.0, std::time::Duration::ZERO);
    let cascade = cascade.expect("the resume certifies");
    assert_eq!(
        cascade.iterations(),
        0,
        "a still-certified point costs no outer iteration"
    );
    assert_eq!(cascade.rho()[0].to_bits(), parent.rho()[0].to_bits());
    assert_eq!(
        resume.recorded(),
        Some(gam_model_api::WarmStartOutcome::Resumed)
    );

    let (seeds, rho, _, iterations) = published_two_basin(&problem);
    assert_eq!(
        seeds,
        vec![parent.rho().clone()],
        "the multistart is the one resumed run"
    );
    assert_eq!((rho.to_bits(), iterations), (parent.rho()[0].to_bits(), 0));
}

/// A point certified for another criterion (a pilot's, an unarmed evidence
/// fit's, an earlier alternation round's) is declined on the parent's own
/// inputs, and the search runs bit for bit as it runs cold. The point here is
/// stationary for this criterion but records another value, so only the value
/// test declines it. The cold cascade publishes the rail and a search from the
/// point would publish the well, so the test catches both a resume that skips
/// the value test and a decline that searches from the point.
#[test]
fn a_prior_certificate_for_another_criterion_is_declined_and_the_search_runs_cold_3002() {
    let parent = two_basin_parent();
    let problem = problem_with_candidates(vec![array![0.0], array![4.0]]);
    let (cold, _) = run_fixture_seed(&problem, 1.0, 6.0, std::time::Duration::ZERO);
    let cold = cold.expect("the cold cascade certifies");
    assert!(
        cold.final_value() > parent.final_value() + 0.5,
        "fixture precondition: the cold cascade publishes the rail (V={}), the point is the \
         well (V={})",
        cold.final_value(),
        parent.final_value()
    );
    let other = warm_start(parent.rho()[0], parent.final_value() + 0.25, true);
    let warm_problem = problem.clone().with_warm_start(&other);
    let (warm, _) = run_fixture_seed(&warm_problem, 1.0, 6.0, std::time::Duration::ZERO);
    let warm = warm.expect("the cold cascade certifies");
    assert_eq!(
        (
            warm.rho()[0].to_bits(),
            warm.final_value().to_bits(),
            warm.iterations()
        ),
        (
            cold.rho()[0].to_bits(),
            cold.final_value().to_bits(),
            cold.iterations()
        ),
        "the declined search ran cold"
    );
    assert!(matches!(
        other.recorded(),
        Some(gam_model_api::WarmStartOutcome::NotUsed(_))
    ));

    let (_, cold_rho, cold_v, _) = published_two_basin(&problem);
    let (seeds, rho, v, _) = published_two_basin(&warm_problem);
    assert!(
        !seeds.contains(parent.rho()),
        "a declined resume adds no seed to the multistart, got {seeds:?}"
    );
    assert_eq!(
        (rho.to_bits(), v.to_bits()),
        (cold_rho.to_bits(), cold_v.to_bits())
    );
}

/// A search that certifies its first certifiable seed does not take a point from
/// other inputs: a new first seed would change which point it certifies. It
/// runs exactly as the cold search and records that it did not use the point.
#[test]
fn a_cascade_does_not_use_a_warm_start_from_other_inputs_3002() {
    let problem = problem_with_candidates(vec![array![0.0], array![4.0]]);
    let (cold, _) = run_fixture_seed(&problem, 1.0, 6.0, std::time::Duration::ZERO);
    let cold = cold.expect("the cold cascade certifies");
    let parent = two_basin_parent();
    let other = warm_start(parent.rho()[0], parent.final_value(), false);
    let (warm, _) = run_fixture_seed(
        &problem.clone().with_warm_start(&other),
        1.0,
        6.0,
        std::time::Duration::ZERO,
    );
    let warm = warm.expect("the cascade certifies");
    assert_eq!(
        (
            warm.rho()[0].to_bits(),
            warm.final_value().to_bits(),
            warm.iterations()
        ),
        (
            cold.rho()[0].to_bits(),
            cold.final_value().to_bits(),
            cold.iterations()
        ),
        "the cascade ran cold"
    );
    assert!(matches!(
        other.recorded(),
        Some(gam_model_api::WarmStartOutcome::NotUsed(_))
    ));
}
