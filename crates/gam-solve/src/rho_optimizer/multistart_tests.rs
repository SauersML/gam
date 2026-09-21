// Child module of `run::multistart` (see the `#[path]` declaration there).
#![cfg(test)]

use super::*;
use ndarray::array;

/// What each seed's run reported, by seed index. A multistart publishes one
/// payload (#3238), so a test that inspects every run records it here.
struct SeedRecords<T>(std::sync::Mutex<Vec<Option<T>>>);

impl<T> SeedRecords<T> {
    fn new(seeds: usize) -> Self {
        Self(std::sync::Mutex::new((0..seeds).map(|_| None).collect()))
    }

    fn record(&self, index: usize, value: T) {
        self.0.lock().expect("records")[index] = Some(value);
    }

    /// Every seed's record, in seed order.
    fn all(self) -> Vec<T> {
        self.0
            .into_inner()
            .expect("records")
            .into_iter()
            .enumerate()
            .map(|(index, value)| value.unwrap_or_else(|| panic!("seed {index} ran")))
            .collect()
    }
}

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

/// The fixture problem started at `own`, with `levels` the declared starts
/// searched beside it.
fn problem_with_starts(own: f64, levels: &[f64]) -> (OuterProblem, Vec<f64>) {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_initial_rho(array![own])
        .with_bounds(array![-8.0], array![8.0]);
    (problem, levels.to_vec())
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
        move |_: &mut Vec<f64>, theta: &Array1<f64>| {
            Ok(railed_and_well(theta[0], side, centre).cost)
        },
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
    let (problem, levels) = problem_with_starts(0.0, &[4.0]);
    let seeds = problem.multistart_seeds(None, &levels).expect("seeds");
    assert!(
        seeds.iter().any(|seed| seed[0] == 0.0) && seeds.iter().any(|seed| seed[0] == 4.0),
        "fixture precondition: both the neutral and the stiff seed are generated, got {seeds:?}"
    );
    let searched = SeedRecords::new(seeds.len());
    let outcome = problem
        .run_certified_multistart(&levels, "stiff control", 0, 1, |index, seed_problem, _| {
            let (outcome, seen) =
                run_fixture_seed(&seed_problem, 1.0, 6.0, std::time::Duration::ZERO);
            searched.record(index, seen.len());
            (outcome, ())
        })
        .expect("multistart runs");
    let winner = outcome.winner.expect("a seed certifies");
    let published = outcome.outcomes[winner]
        .as_ref()
        .expect("the winner certified");
    assert!(
        (published.rho()[0] - 6.0).abs() < 0.1 && published.final_value() < 1.0e-3,
        "published rho={} value={}: the well at rho = 6 must win",
        published.rho()[0],
        published.final_value(),
    );
    assert_eq!(
        outcome.seeds[winner][0], 4.0,
        "the stiff seed reaches the well"
    );
    for (index, (run, evaluations)) in outcome.outcomes.iter().zip(searched.all()).enumerate() {
        assert!(
            run.is_ok(),
            "seed {index} certifies: {:?}",
            run.as_ref().err()
        );
        assert!(evaluations > 0, "seed {index} ran its own search");
    }
}

/// The #2359 shape, mirrored: the neutral seed falls to the upper rail and the
/// flexible ρ = −2 seed, which starts above that railed value, reaches the well
/// at ρ = −5. Searching every seed publishes the well.
#[test]
fn multistart_publishes_the_basin_only_a_flexible_seed_reaches_2359() {
    let (problem, levels) = problem_with_starts(0.0, &[-2.0]);
    let outcome = problem
        .run_certified_multistart(&levels, "flexible control", 0, 1, |_, seed_problem, _| {
            run_fixture_seed(&seed_problem, -1.0, -5.0, std::time::Duration::ZERO)
        })
        .expect("multistart runs");
    let winner = outcome.winner.expect("a seed certifies");
    let published = outcome.outcomes[winner]
        .as_ref()
        .expect("the winner certified");
    assert!(
        (published.rho()[0] + 5.0).abs() < 0.1 && published.final_value() < 1.0e-3,
        "published rho={} value={}: the well at rho = -5 must win",
        published.rho()[0],
        published.final_value(),
    );
}

/// A problem that declares no further start searches its own start alone, and
/// each multistart run receives exactly one seed: the declared levels, each
/// projected into the search box, in declaration order, without duplicates.
#[test]
fn each_multistart_run_searches_its_own_seed_alone_2359() {
    let (problem, levels) = problem_with_starts(0.0, &[4.0, 0.0, f64::INFINITY]);
    assert_eq!(
        problem.multistart_seeds(None, &[]).expect("seeds"),
        vec![array![0.0]],
        "with no declared level the fit's own start is the only search"
    );
    let every = problem.multistart_seeds(None, &levels).expect("seeds");
    assert_eq!(
        every,
        vec![array![0.0], array![4.0], array![8.0]],
        "the own start first, the duplicate dropped, +inf on the upper face"
    );
    let sole = SeedRecords::new(every.len());
    let outcome = problem
        .run_certified_multistart(&levels, "sole seeds", 0, 1, |index, seed_problem, _| {
            sole.record(
                index,
                seed_problem.multistart_seeds(None, &[]).expect("seeds"),
            );
            (
                Err(EstimationError::RemlOptimizationFailed(
                    "not run".to_string(),
                )),
                (),
            )
        })
        .expect("multistart runs");
    assert_eq!(outcome.seeds, every);
    for (index, seeds) in sole.all().into_iter().enumerate() {
        assert_eq!(
            seeds,
            vec![every[index].clone()],
            "run {index} searches its own seed alone"
        );
    }
    assert_eq!(
        outcome.winner, None,
        "no run certified, so there is no winner"
    );
}

/// The winner is a function of the seed runs' results alone, not of the order
/// they finish in: with the finishing order of the seed runs reversed, the same
/// seed wins with the same bits, and it is the seed keep-best picks by value.
#[test]
fn a_multistart_winner_does_not_depend_on_which_run_finishes_first_2359() {
    let (problem, levels) = problem_with_starts(0.0, &[3.5, 4.0]);
    let seeds = problem.multistart_seeds(None, &levels).expect("seeds");
    let mut published = Vec::new();
    for reverse in [false, true] {
        let outcome = problem
            .run_certified_multistart(
                &levels,
                "finishing order",
                0,
                1,
                |index, seed_problem, _| {
                    let rank = if reverse { seeds.len() - index } else { index };
                    run_fixture_seed(
                        &seed_problem,
                        1.0,
                        6.0,
                        std::time::Duration::from_millis(40 * rank as u64),
                    )
                },
            )
            .expect("multistart runs");
        let winner = outcome.winner.expect("a seed certifies");
        let result = outcome.outcomes[winner]
            .as_ref()
            .expect("the winner certified");
        let lowest = outcome
            .outcomes
            .iter()
            .filter_map(|run| run.as_ref().ok().map(CertifiedOuterResult::final_value))
            .fold(f64::INFINITY, f64::min);
        assert!(
            result.final_value() - lowest
                <= outer_value_agreement_bound(result.final_value(), lowest),
            "keep-best publishes the lowest certified value, to its rounding envelope"
        );
        published.push((
            winner,
            result.rho()[0].to_bits(),
            result.final_value().to_bits(),
        ));
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
    let (problem, levels) = problem_with_starts(5.75, &[6.25]);
    let seeds = problem.multistart_seeds(None, &levels).expect("seeds");
    let first_on_floor = seeds
        .iter()
        .position(|seed| (seed[0] - 6.0).abs() < 0.5)
        .expect("fixture precondition: a seed sits on the well's floor");
    for reverse in [false, true] {
        let outcome = problem
            .run_certified_multistart(&levels, "tie", 0, 1, |index, seed_problem, _| {
                let rank = if reverse { seeds.len() - index } else { index };
                run_flat_well_seed(
                    &seed_problem,
                    std::time::Duration::from_millis(40 * rank as u64),
                )
            })
            .expect("multistart runs");
        let on_floor: Vec<usize> = outcome
            .outcomes
            .iter()
            .enumerate()
            .filter(|(_, run)| {
                run.as_ref()
                    .is_ok_and(|certified| certified.final_value() == 0.0)
            })
            .map(|(index, _)| index)
            .collect();
        assert!(
            on_floor.len() >= 2,
            "fixture precondition: at least two exact ties, got {on_floor:?}"
        );
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
    let (problem, levels) = problem_with_starts(0.0, &[4.0]);
    let seeds = problem.multistart_seeds(None, &levels).expect("seeds");
    let stiff = seeds
        .iter()
        .position(|seed| seed[0] == 4.0)
        .expect("fixture precondition: the stiff seed is generated");
    // The stiff seed, the only one that reaches the well, does not certify here.
    let outcome = problem
        .run_certified_multistart(&levels, "certified only", 0, 1, |index, seed_problem, _| {
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
    let published = outcome.outcomes[winner]
        .as_ref()
        .expect("the winner certified");
    assert!(
        published.final_value() > 0.99,
        "the certified rail wins: {}",
        published.final_value()
    );

    let refused = problem
        .run_certified_multistart(&levels, "none certified", 0, 1, |index, _, _| {
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
        let entry = format!(
            "seed {index} rho={:?}: not certified [",
            seeds[index].to_vec()
        );
        let at = message.find(&entry).unwrap_or_else(|| {
            panic!("the refusal carries seed {index}'s start and verdict: {message}")
        });
        assert!(
            at >= previous,
            "the refusal lists seeds in seed order: {message}"
        );
        previous = at;
        assert!(
            message.contains(&format!("planted: seed {index} did not certify")),
            "the refusal carries seed {index}'s outcome: {message}"
        );
    }
    assert_eq!(
        message
            .matches("[EstimationError::RemlOptimizationFailed]")
            .count(),
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
    let (problem, levels) = problem_with_starts(0.0, &[3.5, 4.0]);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .build()
        .expect("a test pool");
    let lanes = SeedRecords::new(3);
    let outcome = pool.install(|| {
        problem
            .run_certified_multistart_on(
                governor,
                &levels,
                "admission",
                PRE_LAUNCH_AVAILABLE,
                working_set_bytes,
                |index, seed_problem, lane| {
                    lanes.record(index, (lane.serial_available_bytes(), lane.granted_bytes()));
                    let (outcome, _) = run_fixture_seed(&seed_problem, 1.0, 6.0, hold);
                    (outcome, ())
                },
            )
            .expect("multistart runs")
    });
    let winner = outcome.winner.expect("a seed certifies");
    let result = outcome.outcomes[winner]
        .as_ref()
        .expect("the winner certified");
    (
        (
            winner,
            result.rho()[0].to_bits(),
            result.final_value().to_bits(),
        ),
        outcome.most_live,
        lanes.all(),
    )
}

/// Memory pressure: a budget that admits one search's working set but not two
/// makes the searches queue, one live at a time, instead of taking a smaller
/// path. A search is granted its full working set, or none: beside a finished
/// run whose payload can still be published, and so still holds its grant
/// (#3238), the budget has no room for a second working set, and the search
/// runs as the serial search. Every search reads the same pre-launch
/// availability, and the winner is bitwise the one a budget admitting all three
/// at once publishes.
#[test]
fn a_budget_for_one_search_queues_the_rest_and_keeps_the_winner_2359() {
    let working_set = 1_000usize;
    let hold = std::time::Duration::from_millis(60);
    let tight =
        gam_runtime::resource::MemoryGovernor::with_budget_bytes(working_set + working_set / 2);
    let (queued_winner, queued_live, queued_lanes) =
        admitted_fixture_run(&tight, working_set, 3, hold);
    assert_eq!(queued_live, 1, "a budget for one search runs one at a time");
    assert!(
        queued_lanes
            .iter()
            .all(|&(_, granted)| granted == working_set || granted == 0)
            && queued_lanes
                .iter()
                .any(|&(_, granted)| granted == working_set),
        "a queued search is granted its full working set or none, never a smaller one, \
         and the first is granted: {queued_lanes:?}"
    );
    let ample = gam_runtime::resource::MemoryGovernor::with_budget_bytes(working_set * 16);
    let (parallel_winner, parallel_live, parallel_lanes) =
        admitted_fixture_run(&ample, working_set, 3, hold);
    assert!(
        parallel_live > 1,
        "an ample budget runs searches at once (most live {parallel_live})"
    );
    assert_eq!(
        queued_winner, parallel_winner,
        "queueing does not move the winner or its bits"
    );
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
    assert!(
        lanes.iter().all(|&(_, granted)| granted == 0),
        "no grant: {lanes:?}"
    );
    let ample = gam_runtime::resource::MemoryGovernor::with_budget_bytes(1_000_000);
    let (winner, _, _) = admitted_fixture_run(&ample, 1_000, 3, std::time::Duration::ZERO);
    assert_eq!(serial_winner, winner);
}

/// Called from outside every Rayon pool the lanes are tasks of gam's process
/// pool; called inside a caller's own pool (the path gnomon's calibrate pool
/// takes) they are that pool's tasks. At pool widths 1, 4 and 12 and on both
/// paths the same seed wins with the same bits, and the pool's width bounds the
/// lanes.
#[test]
fn the_winner_is_the_same_on_every_pool_width_and_pool_2359() {
    let (problem, levels) = problem_with_starts(0.0, &[3.5, 4.0]);
    let run = |problem: &OuterProblem| {
        let outcome = problem
            .run_certified_multistart(&levels, "pool width", 0, 1, |_, seed_problem, _| {
                run_fixture_seed(&seed_problem, 1.0, 6.0, std::time::Duration::ZERO)
            })
            .expect("multistart runs");
        let winner = outcome.winner.expect("a seed certifies");
        let result = outcome.outcomes[winner]
            .as_ref()
            .expect("the winner certified");
        (
            outcome.lanes,
            (
                winner,
                result.rho()[0].to_bits(),
                result.final_value().to_bits(),
            ),
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
        assert_eq!(
            inside, outside,
            "pool width {width}: same winner and bits as on the process pool"
        );
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

/// The published (seeds, ρ, V, iterations) of a multistart on the two-basin
/// fixture, searching `levels` beside the problem's own start.
fn published_two_basin(
    problem: &OuterProblem,
    levels: &[f64],
) -> (Vec<Array1<f64>>, f64, f64, usize) {
    let outcome = problem
        .run_certified_multistart(levels, "two basins", 0, 1, |_, seed_problem, _| {
            run_fixture_seed(&seed_problem, 1.0, 6.0, std::time::Duration::ZERO)
        })
        .expect("multistart runs");
    let winner = outcome.winner.expect("a seed certifies");
    let result = outcome.outcomes[winner]
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
    let (problem, _) = problem_with_starts(4.0, &[]);
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
    let (problem, levels) = problem_with_starts(0.0, &[4.0]);
    let (_, _, cold, _) = published_two_basin(&problem, &levels);
    let (_, alone_rho, alone, _) =
        published_two_basin(&problem.clone().with_initial_rho(array![-6.0]), &[]);
    assert!(
        alone > cold + 0.5,
        "fixture precondition: the parent's point alone publishes the rail \
         (rho={alone_rho} V={alone}), the cold seeds the well (V={cold})"
    );
    let parent = warm_start(-6.0, alone, false);
    let (seeds, warm_rho, warm, _) =
        published_two_basin(&problem.clone().with_warm_start(&parent), &levels);
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
/// the parent's bits. Through a single search and through the multistart alike.
#[test]
fn a_warm_start_on_the_parents_inputs_resumes_with_no_outer_iteration_3002() {
    let parent = two_basin_parent();
    let resume = warm_start(parent.rho()[0], parent.final_value(), true);
    let (problem, levels) = problem_with_starts(0.0, &[4.0]);
    let problem = problem.with_warm_start(&resume);
    let (single, _) = run_fixture_seed(&problem, 1.0, 6.0, std::time::Duration::ZERO);
    let single = single.expect("the resume certifies");
    assert_eq!(
        single.iterations(),
        0,
        "a still-certified point costs no outer iteration"
    );
    assert_eq!(single.rho()[0].to_bits(), parent.rho()[0].to_bits());
    assert_eq!(
        resume.recorded(),
        Some(gam_model_api::WarmStartOutcome::Resumed)
    );

    let (seeds, rho, _, iterations) = published_two_basin(&problem, &levels);
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
/// test declines it. The cold search publishes the rail and a search from the
/// point would publish the well, so the test catches both a resume that skips
/// the value test and a decline that searches from the point.
#[test]
fn a_prior_certificate_for_another_criterion_is_declined_and_the_search_runs_cold_3002() {
    let parent = two_basin_parent();
    let (problem, levels) = problem_with_starts(0.0, &[4.0]);
    let (cold, _) = run_fixture_seed(&problem, 1.0, 6.0, std::time::Duration::ZERO);
    let cold = cold.expect("the cold search certifies");
    assert!(
        cold.final_value() > parent.final_value() + 0.5,
        "fixture precondition: the cold search publishes the rail (V={}), the point is the \
         well (V={})",
        cold.final_value(),
        parent.final_value()
    );
    let other = warm_start(parent.rho()[0], parent.final_value() + 0.25, true);
    let warm_problem = problem.clone().with_warm_start(&other);
    let (warm, _) = run_fixture_seed(&warm_problem, 1.0, 6.0, std::time::Duration::ZERO);
    let warm = warm.expect("the cold search certifies");
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

    let (_, cold_rho, cold_v, _) = published_two_basin(&problem, &levels);
    let (seeds, rho, v, _) = published_two_basin(&warm_problem, &levels);
    assert!(
        !seeds.contains(parent.rho()),
        "a declined resume adds no seed to the multistart, got {seeds:?}"
    );
    assert_eq!(
        (rho.to_bits(), v.to_bits()),
        (cold_rho.to_bits(), cold_v.to_bits())
    );
}

/// A single search, which certifies from its one derived start, does not take a
/// point from other inputs: a new start would change which point it certifies.
/// It runs exactly as the cold search and records that it did not use the point.
#[test]
fn a_single_search_does_not_use_a_warm_start_from_other_inputs_3002() {
    let (problem, _) = problem_with_starts(0.0, &[4.0]);
    let (cold, _) = run_fixture_seed(&problem, 1.0, 6.0, std::time::Duration::ZERO);
    let cold = cold.expect("the cold search certifies");
    let parent = two_basin_parent();
    let other = warm_start(parent.rho()[0], parent.final_value(), false);
    let (warm, _) = run_fixture_seed(
        &problem.clone().with_warm_start(&other),
        1.0,
        6.0,
        std::time::Duration::ZERO,
    );
    let warm = warm.expect("the single search certifies");
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
        "the single search ran cold"
    );
    assert!(matches!(
        other.recorded(),
        Some(gam_model_api::WarmStartOutcome::NotUsed(_))
    ));
}

/// A payload that counts how many of its kind are alive, and names its seed.
struct CountedPayload {
    seed: usize,
    alive: std::sync::Arc<AtomicUsize>,
}

impl CountedPayload {
    fn new(seed: usize, alive: &std::sync::Arc<AtomicUsize>) -> Self {
        alive.fetch_add(1, Ordering::SeqCst);
        Self {
            seed,
            alive: std::sync::Arc::clone(alive),
        }
    }
}

impl Drop for CountedPayload {
    fn drop(&mut self) {
        self.alive.fetch_sub(1, Ordering::SeqCst);
    }
}

/// #3238: a serial multistart holds one finished run's payload at a time, and the
/// ledger carries it. The four seeds run in seed order on one lane: the neutral
/// start certifies on the rail, the ρ = 3.5 start reaches the well and displaces
/// it, and the ρ = 4 and ρ = −2 starts tie with or lose to that incumbent. Holding
/// every payload until the last seed finished, the k-th search started beside k
/// payloads, none of them on the ledger. Keeping only a payload that can still be
/// published, each search starts beside the incumbent's alone, whose grant is still
/// reserved; the published payload is the winner's, and no grant outlives the
/// multistart.
#[test]
fn a_serial_multistart_holds_one_finished_payload_and_charges_it_3238() {
    let (problem, levels) = problem_with_starts(0.0, &[3.5, 4.0, -2.0]);
    let seeds = problem.multistart_seeds(None, &levels).expect("seeds");
    assert_eq!(
        seeds.len(),
        4,
        "fixture precondition: four distinct seeds, got {seeds:?}"
    );
    let working_set = 1_000usize;
    let budget = working_set * 16;
    let governor = gam_runtime::resource::MemoryGovernor::with_budget_bytes(budget);
    let alive = std::sync::Arc::new(AtomicUsize::new(0));
    // (payloads alive, bytes reserved) as each search starts.
    let at_start = SeedRecords::new(seeds.len());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("a test pool");
    let outcome = pool.install(|| {
        problem
            .run_certified_multistart_on(
                &governor,
                &levels,
                "held payloads",
                PRE_LAUNCH_AVAILABLE,
                working_set,
                |index, seed_problem, _| {
                    at_start.record(
                        index,
                        (
                            alive.load(Ordering::SeqCst),
                            budget - governor.remaining_bytes(),
                        ),
                    );
                    let (outcome, _) =
                        run_fixture_seed(&seed_problem, 1.0, 6.0, std::time::Duration::ZERO);
                    (outcome, CountedPayload::new(index, &alive))
                },
            )
            .expect("multistart runs")
    });
    let winner = outcome.winner.expect("a seed certifies");
    assert_eq!(
        outcome.lanes, 1,
        "fixture precondition: one lane runs the seeds in order"
    );
    let at_start = at_start.all();
    for (index, &(held, reserved)) in at_start.iter().enumerate().skip(1) {
        assert_eq!(
            held, 1,
            "search {index} started beside {held} finished payloads; only the incumbent's can \
             still be published: {at_start:?}"
        );
        assert_eq!(
            reserved,
            working_set * (1 + held),
            "search {index}: the ledger carries its own grant and the held payload's: {at_start:?}"
        );
    }
    assert_eq!(
        outcome.payload.seed, winner,
        "the published payload is the winner's"
    );
    assert_eq!(
        alive.load(Ordering::SeqCst),
        1,
        "only the published payload outlives the multistart"
    );
    assert_eq!(
        governor.remaining_bytes(),
        budget,
        "no grant outlives the multistart"
    );
}

/// Every assignment of `values` to `seeds` runs, one value per run (`None`: the
/// run did not certify).
fn every_state(seeds: usize, values: &[Option<f64>]) -> Vec<Vec<Option<f64>>> {
    let mut states = vec![Vec::new()];
    for _ in 0..seeds {
        states = states
            .into_iter()
            .flat_map(|state| {
                values.iter().map(move |&value| {
                    let mut next = state.clone();
                    next.push(value);
                    next
                })
            })
            .collect();
    }
    states
}

/// Every order in which `seeds` runs can finish.
fn finishing_orders(seeds: usize) -> Vec<Vec<usize>> {
    if seeds == 0 {
        return vec![Vec::new()];
    }
    let mut orders = Vec::new();
    for order in finishing_orders(seeds - 1) {
        for at in 0..=order.len() {
            let mut next = order.clone();
            next.insert(at, seeds - 1);
            orders.push(next);
        }
    }
    orders
}

/// #3238: dropping a finished run's payload is safe, and in seed order it is
/// complete. Over every outcome of four seeds (not certified, or certified at 0,
/// at a value tying 1 within its rounding envelope, at 1, or at 2) and every order
/// the runs can finish in, the payload a multistart publishes (the winner's, or
/// the first seed's when none certifies) is publishable in every intermediate
/// state; and when the runs finish in seed order, at most one finished payload is
/// ever held.
#[test]
fn only_a_payload_that_cannot_be_published_is_dropped_3238() {
    let tie = 1.0 + outer_value_agreement_bound(1.0, 1.0) / 2.0;
    let values = [None, Some(0.0), Some(1.0), Some(tie), Some(2.0)];
    assert!(
        !displaces(1.0, tie) && !displaces(tie, 1.0) && displaces(1.0, 0.0),
        "fixture precondition: 1 and {tie} tie, 0 displaces 1"
    );
    let orders = finishing_orders(4);
    for outcome in every_state(4, &values) {
        let published = keep_best(outcome.iter().copied()).unwrap_or(0);
        for order in &orders {
            let mut state: Vec<Option<Option<f64>>> = vec![None; outcome.len()];
            for &finished in order {
                state[finished] = Some(outcome[finished]);
                let publishable = publishable_payloads(&state);
                assert!(
                    publishable[published],
                    "outcomes {outcome:?} finishing in order {order:?}: the published seed \
                     {published} was ruled out at {state:?}"
                );
                if order.iter().zip(0..).all(|(&seed, index)| seed == index) {
                    let held = state
                        .iter()
                        .zip(&publishable)
                        .filter(|(value, keep)| value.is_some() && **keep)
                        .count();
                    assert!(
                        held <= 1,
                        "outcomes {outcome:?} in seed order hold {held} payloads at {state:?}"
                    );
                }
            }
        }
    }
}

/// `V = 3 + ½(ρ − 2)²`, with its analytic gradient and Hessian: one convex well,
/// so every seed certifies the same optimum and every local model's minimum is
/// that optimum's value.
fn convex_well(rho: f64) -> OuterEval {
    OuterEval {
        cost: 3.0 + 0.5 * (rho - 2.0).powi(2),
        gradient: array![rho - 2.0],
        hessian: HessianValue::Dense(array![[1.0]]),
        inner_beta_hint: None,
    }
}

/// The convex-well problem at `n` observations (`None`: the route declares none),
/// started at `own`, with the analytic Hessian the ARC route certifies with.
fn convex_well_problem(own: f64, n: Option<usize>) -> OuterProblem {
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_initial_rho(array![own])
        .with_bounds(array![-8.0], array![8.0]);
    match n {
        Some(n) => problem.with_problem_size(n, 1),
        None => problem,
    }
}

/// Run one convex-well seed, counting its evaluations. `hold` makes every
/// evaluation wait until the seed's multistart has a confirmed quorum.
fn run_convex_well_seed(
    problem: &OuterProblem,
    hold: bool,
) -> (Result<CertifiedOuterResult, EstimationError>, usize) {
    let release = problem.seed_release.clone();
    let wait = move || {
        if !hold {
            return;
        }
        let quorum = &release.as_ref().expect("a held seed has a release handle").quorum;
        assert!(
            quorum.wait_floor(std::time::Duration::from_secs(120)).is_some(),
            "the two fast seeds never formed a quorum"
        );
    };
    let wait_value = wait.clone();
    let mut obj = problem.build_objective(
        0usize,
        move |evaluations: &mut usize, theta: &Array1<f64>| {
            wait_value();
            *evaluations += 1;
            Ok(convex_well(theta[0]).cost)
        },
        move |evaluations: &mut usize, theta: &Array1<f64>| {
            wait();
            *evaluations += 1;
            Ok(convex_well(theta[0]))
        },
        None::<fn(&mut usize)>,
        None::<fn(&mut usize, &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let outcome = problem.run_certified(&mut obj, "convex well");
    (outcome, obj.state)
}

/// #3325: once two seeds certify one optimum, a seed still searching whose
/// evidence cannot reach below it is released at its next evaluation, and the
/// published run is the one keep-best picks among the certified seeds.
#[test]
fn a_certified_quorum_releases_a_seed_that_cannot_beat_it_3325() {
    let problem = convex_well_problem(0.0, Some(200));
    let levels = [1.0, 5.0];
    let seeds = problem.multistart_seeds(None, &levels).expect("seeds");
    assert_eq!(seeds, vec![array![0.0], array![1.0], array![5.0]]);
    let evaluations = SeedRecords::new(seeds.len());
    let outcome = problem
        .run_certified_multistart(&levels, "quorum release", 0, 1, |index, seed_problem, _| {
            assert!(
                seed_problem.seed_release.is_some(),
                "a route with an observation count gives every seed a release handle"
            );
            let (outcome, count) = run_convex_well_seed(&seed_problem, index == 2);
            evaluations.record(index, count);
            (outcome, ())
        })
        .expect("multistart runs");
    let evaluations = evaluations.all();
    let winner = outcome.winner.expect("the fast seeds certify");
    assert!(winner < 2, "a fast seed wins, got seed {winner}");
    let published = outcome.outcomes[winner].as_ref().expect("the winner certified");
    assert!(
        (published.rho()[0] - 2.0).abs() < 1.0e-3 && (published.final_value() - 3.0).abs() < 1.0e-6,
        "published rho={} value={}",
        published.rho()[0],
        published.final_value(),
    );
    for index in 0..2 {
        assert!(
            outcome.outcomes[index].is_ok(),
            "fast seed {index} certifies: {:?}",
            outcome.outcomes[index].as_ref().err()
        );
    }
    let released = outcome.outcomes[2]
        .as_ref()
        .err()
        .expect("the held seed is released, not certified");
    assert!(
        released.is_fatal_outer_evaluation() && released.to_string().contains("released"),
        "the held seed ends on its release: {released}"
    );
    assert!(
        evaluations[2] < evaluations[0].min(evaluations[1]),
        "the released seed stops short of a full search: evaluations {evaluations:?}"
    );
}

/// A route that declares no observation count has no resolution to judge one
/// optimum at, so its multistart releases nothing and every seed certifies.
#[test]
fn a_multistart_without_an_observation_count_releases_no_seed_3325() {
    let problem = convex_well_problem(0.0, None);
    let outcome = problem
        .run_certified_multistart(&[1.0, 5.0], "no quorum", 0, 1, |_, seed_problem, _| {
            assert!(seed_problem.seed_release.is_none());
            run_convex_well_seed(&seed_problem, false)
        })
        .expect("multistart runs");
    for (index, run) in outcome.outcomes.iter().enumerate() {
        assert!(run.is_ok(), "seed {index} certifies: {:?}", run.as_ref().err());
    }
}

/// Whether a seed guarded by a quorum whose floor is `floor` (`None`: no quorum
/// yet) is admitted to its next evaluation after evaluating `script` in order.
fn admitted_after(floor: Option<f64>, script: Vec<OuterEval>) -> bool {
    let quorum = Arc::new(SeedQuorum::new(1.0e-3));
    if let Some(floor) = floor {
        quorum.floor.store(floor.to_bits(), Ordering::Release);
    }
    let handle = SeedReleaseHandle { quorum, seed: 7 };
    let problem = convex_well_problem(0.0, Some(500));
    let mut script = std::collections::VecDeque::from(script);
    let steps = script.len();
    let mut inner = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(convex_well(theta[0]).cost),
        move |_: &mut (), theta: &Array1<f64>| {
            Ok(script.pop_front().unwrap_or_else(|| convex_well(theta[0])))
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut guarded = handle.guard(&mut inner, (array![-8.0], array![8.0]), "release guard");
    for step in 0..steps {
        guarded
            .eval(&array![0.0])
            .unwrap_or_else(|error| panic!("scripted evaluation {step} is admitted: {error}"));
    }
    match guarded.eval(&array![0.0]) {
        Ok(_) => true,
        Err(error) => {
            assert!(error.is_fatal_outer_evaluation(), "a release is fatal: {error}");
            assert!(
                guarded.eval_cost(&array![0.0]).is_err(),
                "a released seed stays released"
            );
            false
        }
    }
}

fn scripted(cost: f64, gradient: f64, hessian: HessianValue) -> OuterEval {
    OuterEval {
        cost,
        gradient: array![gradient],
        hessian,
        inner_beta_hint: None,
    }
}

/// #3325: a seed is released only when its lowest value and its local model's
/// minimum over the feasible box `[-8, 8]`, on an analytic Hessian, both stay
/// above the quorum's floor. Every scripted evaluation is at `ρ = 0`.
#[test]
fn a_seed_is_released_only_when_its_own_evidence_stays_above_the_quorum_3325() {
    let dense = |h: f64| HessianValue::Dense(array![[h]]);
    // Model minimum 1.5 − ½·0.01 = 1.495 above the floor: released.
    assert!(!admitted_after(Some(1.0), vec![scripted(1.5, 0.1, dense(1.0))]));
    // No quorum yet: never released.
    assert!(admitted_after(None, vec![scripted(1.5, 0.1, dense(1.0))]));
    // Model minimum 1.5 − ½·4 = −0.5 below the floor: keeps searching.
    assert!(admitted_after(Some(1.0), vec![scripted(1.5, 2.0, dense(1.0))]));
    // An evaluated value below the floor: keeps searching.
    assert!(admitted_after(
        Some(1.0),
        vec![scripted(0.5, 0.0, dense(1.0)), scripted(1.5, 0.1, dense(1.0))]
    ));
    // Negative curvature: the model falls to 1.5 + 0.8 − 32 at the box's face,
    // below the floor, so it keeps searching.
    assert!(admitted_after(Some(1.0), vec![scripted(1.5, 0.1, dense(-1.0))]));
    // Slight negative curvature over the box: the model's least value is
    // 1.5 − 0.08 − 0.032 = 1.388, above the floor, so it is released.
    assert!(!admitted_after(Some(1.0), vec![scripted(1.5, 0.01, dense(-1.0e-3))]));
    // No analytic Hessian bounds nothing: keeps searching.
    assert!(admitted_after(
        Some(1.0),
        vec![scripted(1.5, 0.1, HessianValue::Unavailable)]
    ));
    // Its incumbent's model reaches below the floor, though its latest does not.
    assert!(admitted_after(
        Some(1.0),
        vec![scripted(1.2, 1.0, dense(1.0)), scripted(1.5, 0.1, dense(1.0))]
    ));
    // Not one evaluation with derivatives yet: keeps searching.
    assert!(admitted_after(Some(1.0), Vec::new()));
}

/// #3325: two certified runs are one optimum when their values and both
/// Hessians' `½|δᵀHδ|` lie within the certificate's tolerance; equal values in
/// two separate basins are not, nor is a certificate without a Hessian.
#[test]
fn a_quorum_needs_one_optimum_not_one_value_3325() {
    let member = |value: f64, rho: f64, hessian: Option<f64>| QuorumMember {
        index: 0,
        value,
        rho: array![rho],
        hessian: hessian.map(|h| array![[h]]),
    };
    let tau = 1.0 / 400.0;
    let optimum = member(3.0, 2.0, Some(1.0));
    let tolerance = same_optimum(&optimum, &member(3.0 + 1.0e-4, 2.05, Some(1.0)), tau)
        .expect("a nearby certificate of the same well confirms it");
    assert!((tolerance - tau).abs() < 1.0e-6, "tolerance {tolerance} is max(τ − b, b)");
    assert_eq!(same_optimum(&optimum, &member(3.0, 4.0, Some(1.0)), tau), None);
    assert_eq!(same_optimum(&optimum, &member(3.01, 2.0, Some(1.0)), tau), None);
    assert_eq!(same_optimum(&optimum, &member(3.0, 2.0, None), tau), None);
}

/// Under `parallel::install` every lane is a pool task, so a worker waiting at a
/// join inside a live search can steal a lane. An admission that waited for that
/// search's grant would then block the worker the search needs: a lane refused
/// beside a live search must retire at once and be respawned by the release.
#[test]
fn a_lane_refused_beside_a_live_search_retires_at_once_instead_of_waiting_2359() {
    static GOVERNOR: std::sync::OnceLock<gam_runtime::resource::MemoryGovernor> =
        std::sync::OnceLock::new();
    let working_set = 1_000usize;
    let governor = GOVERNOR.get_or_init(|| {
        gam_runtime::resource::MemoryGovernor::with_budget_bytes(working_set + working_set / 2)
    });
    let admission = std::sync::Arc::new(LaneAdmission {
        governor,
        working_set_bytes: working_set,
        serial_available_bytes: PRE_LAUNCH_AVAILABLE,
        lanes: std::sync::Mutex::new(LaneCount::default()),
        most_live: AtomicUsize::new(0),
    });
    let live = admission
        .admit("admission deadlock pin")
        .expect("the first search is admitted");
    let (sender, receiver) = std::sync::mpsc::channel();
    let refused = std::sync::Arc::clone(&admission);
    std::thread::spawn(move || {
        sender
            .send(refused.admit("admission deadlock pin").is_none())
            .expect("send");
    });
    let retired = receiver
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("an admission refused beside a live search returns instead of waiting for it");
    assert!(retired, "a lane refused while a search is live retires");
    assert_eq!(
        admission.release(vec![(None::<()>, live)]),
        1,
        "the release respawns it"
    );
    let respawned = admission
        .admit("admission deadlock pin")
        .expect("the respawned lane is admitted");
    assert_eq!(respawned.granted_bytes(), working_set);
    assert_eq!(admission.release(vec![(None::<()>, respawned)]), 0);
}
