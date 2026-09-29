//! Known-answer behaviours for causal-state extraction: Dyck-1 (a counter), mod-3 counting
//! (a periodic counter or its three-state cycle) and Dyck-2 (a counter and a top-of-stack
//! register), each drawn from its true process so the readout is the process's own next-token
//! law.

use super::*;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const BOS: u32 = 0;

/// Draw `sequences` sequences of `length` tokens after a BOS from `law`, which maps the
/// generator state and returns the next-token distribution over tokens `1..=R`; `advance`
/// moves the state by a token. The readout row after each prefix is `law` at that prefix.
fn draw<S: Clone>(
    sequences: usize,
    length: usize,
    start: S,
    law: impl Fn(&S) -> Vec<f64>,
    advance: impl Fn(&mut S, u32),
    seed: u64,
) -> Harvest {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tokens = Vec::new();
    let mut starts = vec![0];
    let mut rows = Vec::new();
    let mut readout = 0;
    for _sequence in 0..sequences {
        let mut state = start.clone();
        let mut token = BOS;
        for _position in 0..=length {
            advance(&mut state, token);
            tokens.push(token);
            let probabilities = law(&state);
            readout = probabilities.len();
            let draw: f64 = rng.random_range(0.0..1.0);
            let mut cumulative = 0.0;
            token = probabilities.len() as u32;
            for (index, probability) in probabilities.iter().enumerate() {
                cumulative += probability;
                if draw < cumulative {
                    token = index as u32 + 1;
                    break;
                }
            }
            rows.extend(probabilities);
        }
        starts.push(tokens.len());
    }
    let probabilities = Array2::from_shape_vec((tokens.len(), readout), rows).expect("rows match tokens");
    Harvest::new(readout + 1, 1, tokens, starts, probabilities).expect("valid harvest")
}

/// Dyck-1 over `(` = 1, `)` = 2: at depth 0 a close is rare, otherwise both are even.
fn dyck1(sequences: usize, length: usize, seed: u64) -> Harvest {
    draw(
        sequences,
        length,
        0_i64,
        |&depth| if depth == 0 { vec![0.97, 0.03] } else { vec![0.5, 0.5] },
        // BOS (token 0) leaves the depth unchanged.
        |depth, token| *depth = (*depth + i64::from(token == 1) - i64::from(token == 2)).max(0),
        seed,
    )
}

/// Tokens `a` = 1, `b` = 2; the law depends on the count of `a` modulo 3.
fn mod3(sequences: usize, length: usize, seed: u64) -> Harvest {
    draw(
        sequences,
        length,
        0_i64,
        |&count| match count % 3 {
            0 => vec![0.8, 0.2],
            1 => vec![0.3, 0.7],
            _ => vec![0.5, 0.5],
        },
        |count, token| {
            if token == 1 {
                *count += 1;
            }
        },
        seed,
    )
}

/// Dyck-2 over `(` = 1, `)` = 2, `[` = 3, `]` = 4 with a stack: the matching close is likely,
/// the other close is rare.
fn dyck2(sequences: usize, length: usize, seed: u64) -> Harvest {
    draw(
        sequences,
        length,
        Vec::<u32>::new(),
        |stack| match stack.last() {
            None => vec![0.485, 0.015, 0.485, 0.015],
            Some(1) => vec![0.25, 0.48, 0.25, 0.02],
            Some(_) => vec![0.25, 0.02, 0.25, 0.48],
        },
        // Opens push, closes pop, BOS leaves the stack unchanged.
        |stack, token| {
            if token == 1 || token == 3 {
                stack.push(token);
            } else if token == 2 || token == 4 {
                stack.pop();
            }
        },
        seed,
    )
}

#[test]
fn code_round_trips_and_accounts_its_length() {
    let harvest = dyck1(60, 12, 1);
    let start = distinct_start(&harvest).expect("start");
    let mut structure = start.clone();
    structure.counters.push(Counter { increments: vec![0, 1, -1], period: None });
    structure.counters.push(Counter { increments: vec![0, 1, 0], period: Some(3) });
    structure.registers.push(Register { counter: 0, writes: vec![1] });
    let fitted = fit(&structure, &harvest).expect("fit");
    let message = fitted.machine.encode(harvest.vocabulary(), harvest.readout()).expect("encode");
    assert_eq!(message.len_bits(), fitted.score.machine_bits, "accounted bits equal the message");
    let decoded = Machine::decode(&message, harvest.vocabulary(), harvest.readout()).expect("decode");
    assert_eq!(decoded, fitted.machine);
    let (score, divergences) = fitted.machine.score(&harvest).expect("score");
    assert_eq!(score.machine_bits, fitted.score.machine_bits);
    assert!((score.data_bits - fitted.score.data_bits).abs() <= 1e-6 * (1.0 + score.data_bits));
    assert_eq!(divergences.len(), harvest.rows());
}

#[test]
fn symbol_merge_keeps_the_default_unlisted() {
    let map = SymbolMap::new(4, vec![(5, 1), (7, 2), (9, 3)]).expect("map");
    let (merged, index) = map.merge(2, 0).expect("merge");
    assert_eq!(merged.symbols(), 3);
    assert_eq!(merged.symbol(7), 0, "the kept symbol becomes the default");
    assert_eq!(merged.symbol(11), 0);
    assert_eq!(index[0], 0);
    assert_eq!(index[2], 0);
    assert_ne!(merged.symbol(5), merged.symbol(9));
}

/// Dyck-1 floors the depth at zero, so its state is a reflected walk, not a counter: a
/// close at depth 0 leaves the depth at 0. The search expresses the reflection with a
/// register over the open-minus-close counter: the depth is 0 exactly when no open has left
/// the counter at its current value.
#[test]
fn dyck1_is_a_reflected_counter_and_beats_every_finite_machine() {
    let harvest = dyck1(400, 24, 2);
    let start = distinct_start(&harvest).expect("start");
    let full = search(&start, &harvest, Language::FULL).expect("full search");
    let finite = search(&start, &harvest, Language::FINITE).expect("finite search");
    let machine = &full.fitted.machine;
    let structure = &machine.structure;
    assert_eq!(structure.counters.len(), 1, "{structure:?}");
    let counter = &structure.counters[0];
    assert_eq!(counter.period, None);
    let open = structure.symbols.symbol(1);
    let close = structure.symbols.symbol(2);
    assert_ne!(open, close);
    assert_eq!(counter.increments[open], -counter.increments[close]);
    assert_eq!(counter.increments[open].abs(), 1);
    assert_eq!(structure.registers, vec![Register { counter: 0, writes: vec![open] }], "{structure:?}");
    assert!(full.fitted.score.total() < finite.fitted.score.total());
    // The process law is exact at every cell, so only lattice rounding costs data bits.
    assert!(full.fitted.score.data_bits < 0.01 * harvest.rows() as f64, "{:?}", full.fitted.score);
}

#[test]
fn mod3_counting_is_a_three_cycle() {
    let harvest = mod3(300, 20, 3);
    let start = distinct_start(&harvest).expect("start");
    let full = search(&start, &harvest, Language::FULL).expect("search");
    let predicted = full.fitted.machine.predict(&harvest).expect("predict");
    let mut worst = 0.0_f64;
    for (model, machine) in harvest.probabilities().rows().into_iter().zip(predicted.rows()) {
        for (p, q) in model.iter().zip(machine) {
            worst = worst.max((p - q).abs());
        }
    }
    assert!(worst < 1e-2, "worst readout gap {worst}: {:?}", full.fitted.machine.structure);
}

#[test]
fn dyck2_is_a_counter_with_a_top_of_stack_register() {
    let harvest = dyck2(500, 24, 4);
    let start = distinct_start(&harvest).expect("start");
    let full = search(&start, &harvest, Language::FULL).expect("full search");
    let counters = search(&start, &harvest, Language::COUNTERS).expect("counter search");
    let machine = &full.fitted.machine;
    assert_eq!(machine.kind(), "counter+register", "{:?}", machine.structure);
    assert!(full.fitted.score.total() < counters.fitted.score.total());
    assert!(full.fitted.score.data_bits < 0.01 * harvest.rows() as f64, "{:?}", full.fitted.score);
}

/// The planted mod-3 machine: symbol 1 is `a`, everything else symbol 0; the class counts
/// `a` modulo 3, or modulo `classes` when a coarser machine is wanted.
fn cycle(classes: usize) -> Structure {
    let symbols = SymbolMap::new(2, vec![(1, 1)]).expect("map");
    let table = (0..classes).flat_map(|class| [class, (class + 1) % classes]).collect();
    Structure { symbols, classes, initial: 0, table, counters: Vec::new(), registers: Vec::new() }
}

#[test]
fn the_planted_quotient_is_consistent_and_a_coarser_one_is_not() {
    let harvest = mod3(200, 16, 5);
    let exact = fit(&cycle(3), &harvest).expect("planted fit");
    let exact_check = consistency(&exact.machine, &harvest).expect("consistency");
    assert!(exact_check.defect_bits < 1e-9, "{exact_check:?}");
    assert_eq!(exact_check.cells, 3);
    let coarse = fit(&cycle(2), &harvest).expect("coarse fit");
    let coarse_check = consistency(&coarse.machine, &harvest).expect("consistency");
    assert!(coarse_check.defect_bits > 10.0, "{coarse_check:?}");
    let (row, cell, kl) = coarse_check.worst.expect("a witness");
    assert!(kl > 0.0);
    // The witness row sits in a cell whose rows the model reads differently.
    let cells = coarse.machine.cells(&harvest).expect("cells");
    assert_eq!(cells[row], cell);
    let distinct = (0..harvest.rows())
        .filter(|&other| cells[other] == cell)
        .any(|other| harvest.probabilities().row(other) != harvest.probabilities().row(row));
    assert!(distinct);
    // The defect is the least data term of the cells: the fitted readout pays at least it.
    assert!(coarse.score.data_bits >= coarse_check.defect_bits - 1e-6);
}

/// Mod-3 counting whose sampled sequences read at most one `a`: the harvest visits counts
/// 0 and 1 only, and count 2 is reached by a one-token branch alone.
fn mod3_one_a(sequences: usize, length: usize, seed: u64) -> (Harvest, Vec<Branch>) {
    let law = |count: usize| match count % 3 {
        0 => vec![0.8, 0.2],
        1 => vec![0.3, 0.7],
        _ => vec![0.5, 0.5],
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tokens = Vec::new();
    let mut starts = vec![0];
    let mut rows = Vec::new();
    let mut branches = Vec::new();
    for _sequence in 0..sequences {
        let mut count = 0;
        let mut token = BOS;
        for _position in 0..=length {
            count += usize::from(token == 1);
            tokens.push(token);
            rows.extend(law(count));
            for next in [1_u32, 2] {
                branches.push(Branch { row: tokens.len() - 1, token: next, probabilities: law(count + usize::from(next == 1)) });
            }
            let a_allowed = count == 0 && rng.random_range(0.0..1.0) < 0.3;
            token = if a_allowed { 1 } else { 2 };
        }
        starts.push(tokens.len());
    }
    let probabilities = Array2::from_shape_vec((tokens.len(), 2), rows).expect("rows");
    (Harvest::new(3, 1, tokens, starts, probabilities).expect("harvest"), branches)
}

#[test]
fn refinement_admits_the_unvisited_state_and_certifies_the_pool() {
    let (harvest, pool) = mod3_one_a(200, 12, 6);
    let start = distinct_start(&harvest).expect("start");
    let refined = refine(&start, &harvest, &pool, Language::FULL).expect("refine");
    let everything = harvest.with_branches(&pool).expect("branches");
    let first = consistency(&refined.rounds[0].fitted.machine, &everything).expect("first");
    let last = consistency(&refined.report.fitted.machine, &everything).expect("last");
    assert!(refined.rounds.len() >= 2, "no counterexample was admitted");
    assert!(!refined.admitted.is_empty());
    assert!(first.defect_bits > 50.0, "{first:?}");
    assert!(last.defect_bits < 1e-6, "{last:?}: {:?}", refined.report.fitted.machine.structure);
    // On the whole pool, the refined machine's rounded readout codes the rows in fewer bits
    // than any readout of the first machine's cells could.
    let (score, _) = refined.report.fitted.machine.score(&everything).expect("score");
    assert!(score.data_bits < first.defect_bits, "{score:?} against {first:?}");
}

#[test]
fn a_state_intervention_compiles_to_a_native_edit() {
    use super::super::lift::{TieOrientation, UseMap};
    use super::super::compile::ControlRealization;
    use super::super::supports::EvidenceStatus;
    // A planted linear network: the writer input of a row is its state's code, the writer
    // adds `W a` into a residual that is the state's embedding plus a shared offset.
    let mut rng = StdRng::seed_from_u64(7);
    let (states, inputs_width, width) = (3, 6, 5);
    let codes = Array2::from_shape_fn((states, inputs_width), |_| rng.random_range(-1.0..1.0));
    let writer = Array2::from_shape_fn((width, inputs_width), |_| rng.random_range(-1.0..1.0));
    let offset: Vec<f64> = (0..width).map(|_| rng.random_range(-1.0..1.0)).collect();
    let labels: Vec<usize> = (0..30).map(|row| row % states).collect();
    let inputs = Array2::from_shape_fn((labels.len(), inputs_width), |(row, column)| codes[[labels[row], column]]);
    let mut residual = inputs.dot(&writer.t());
    for mut row in residual.rows_mut() {
        row.iter_mut().zip(&offset).for_each(|(value, shift)| *value += shift);
    }
    let mut registry = TensorRegistry::default();
    registry.register_storage(TensorId("w".into()), writer.view().into_dyn()).expect("storage");
    registry
        .register_use_site(UseSiteId("w#0".into()), TensorId("w".into()), UseMap::Linear(TieOrientation::Identity))
        .expect("use");
    let intervention = StateIntervention {
        registry: &registry,
        storage: TensorId("w".into()),
        site: UseSiteId("w#0".into()),
        native: writer.view(),
        inputs: inputs.view(),
        residual: residual.view(),
        labels: &labels,
    };
    let report = intervention.compile(0, 2, MemoryGovernor::global()).expect("compile");
    // The requirement names the state means only, so the edit is validated on them.
    assert!(
        matches!(report.compiled.realization, ControlRealization::EmpiricallyValidated { .. }),
        "{:?}",
        report.compiled.realization
    );
    let plan = report.compiled.plan.as_ref().expect("a plan");
    let edit = &plan.edits()[0].delta;
    let edited = &residual + &inputs.dot(&edit.right()).dot(&edit.left().t());
    for (row, &label) in labels.iter().enumerate() {
        let expected = if label == 0 { residual.row(2) } else { residual.row(row) };
        for (a, b) in edited.row(row).iter().zip(expected) {
            assert!((a - b).abs() < 1e-9, "row {row} state {label}: {a} vs {b}");
        }
    }
    let damage = report.off_target_damage.expect("damage");
    match damage {
        EvidenceStatus::Exact { value, .. } => assert!(value < 1e-9, "{value}"),
        other => panic!("{other:?}"),
    }
}
