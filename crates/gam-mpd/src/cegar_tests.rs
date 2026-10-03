#![cfg(test)]
//! The refinement verifier: reverse-mode input gradients against finite differences through every
//! node kind, the ascent against enumeration, and the loop's termination on finite domains.

use std::sync::Arc;
use super::cegar::{
    AscentStop, Input, InputDomain, SlotDomain, SlotValue, ascend, exhaustive_worst, family_of, inlined, kl_gradients, ladder,
};
use super::contract::{Contract, FamilyKind};
use super::engine::{Budget, Coarsen, DropBlocks, Primitive, decompose_refined};
use super::operator_program::{
    Basis, Coefficient, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram,
    Provenance, Rule, Scale, Slot, exact_precision,
};
use ndarray::{Array1, Array2};

const TOKENS: usize = 5;
const RAW: usize = 3;
const WIDTH: usize = 4;
const CLASSES: usize = 6;

fn dense(name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>) -> Operator {
    let precision = exact_precision(values.iter().copied()).expect("precision");
    Operator::dense(name, rows.clone(), cols.clone(), values, precision, Provenance::native(name)).expect("dense")
}

fn table(rows: usize, cols: usize, seed: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| ((i * 7 + j * 3) as f64 * 0.61 + seed).sin() * 0.8)
}

/// A program with every node kind: a token slot (or, with `token_as_raw`, the same slot fed as a
/// raw one-hot row), a raw slot, an embedding, a GELU, two-key softmax attention mixing two values,
/// a Hadamard product, an outer product, a concatenation and a readout.
fn every_kind(seed: f64, token_as_raw: bool) -> OperatorProgram {
    let declarations = Declarations {
        domains: vec![Domain { size: TOKENS, cycle: None }, Domain { size: CLASSES, cycle: None }],
        slots: vec![
            if token_as_raw { Slot::Raw { width: TOKENS } } else { Slot::Token { domain: 0 } },
            Slot::Raw { width: RAW },
        ],
        parameters: 0,
    };
    let tokens = if token_as_raw { Interface::native(TOKENS) } else { Interface::uniform(TOKENS, 1, LabelKind::Token, 0) }
        .expect("interface");
    let raw = Interface::native(RAW).expect("interface");
    let model = Interface::native(WIDTH).expect("interface");
    let classes = Interface::uniform(CLASSES, 1, LabelKind::Token, 0).expect("interface");
    let mut operators = vec![
        dense("E", &model, &tokens, table(WIDTH, TOKENS, seed)),
        dense("R", &model, &raw, table(WIDTH, RAW, seed + 1.0)),
        dense("Q", &model, &model, table(WIDTH, WIDTH, seed + 2.0)),
        dense("K1", &model, &model, table(WIDTH, WIDTH, seed + 3.0)),
        dense("K2", &model, &model, table(WIDTH, WIDTH, seed + 4.0)),
        dense("V1", &model, &model, table(WIDTH, WIDTH, seed + 5.0)),
        dense("V2", &model, &model, table(WIDTH, WIDTH, seed + 6.0)),
    ];
    let first = if token_as_raw { Node::Raw { slot: 0 } } else { Node::Feature { slot: 0, basis: 0 } };
    let nodes = vec![
        first,
        Node::Raw { slot: 1 },
        Node::Affine { terms: vec![(0, 0), (1, 1)], bias: None },
        Node::Pointwise { input: 2, laws: vec![Law::Gelu; model.group_count()] },
        Node::Affine { terms: vec![(3, 2)], bias: None },
        Node::Affine { terms: vec![(3, 3)], bias: None },
        Node::Bilinear { left: 4, right: 5, scale: Scale::InverseSqrt(WIDTH as u32) },
        Node::Affine { terms: vec![(3, 4)], bias: None },
        Node::Bilinear { left: 4, right: 7, scale: Scale::InverseSqrt(WIDTH as u32) },
        Node::Softmax { scores: vec![6, 8] },
        Node::Affine { terms: vec![(3, 5)], bias: None },
        Node::Affine { terms: vec![(3, 6)], bias: None },
        Node::Mix { weights: 9, payloads: vec![(0, 10), (1, 11)] },
        Node::Hadamard { left: 12, right: 3 },
        Node::Outer { left: 13, right: 1 },
        Node::Concat { parts: vec![13, 14] },
    ];
    let mut program = OperatorProgram {
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }, Basis::Indicator { domain: 1 }],
        operators: operators.iter().cloned().map(Arc::new).collect(),
        rules: Vec::new(),
        nodes: nodes.clone(),
        output: 15,
    };
    let joined = program.node_interface(15).expect("concat interface");
    operators.push(dense("U", &classes, &joined, table(CLASSES, joined.width(), seed + 7.0)));
    program.operators = operators.into_iter().map(Arc::new).collect();
    program.nodes.push(Node::Affine { terms: vec![(15, 7)], bias: None });
    program.nodes.push(Node::Readout { input: 16, basis: 1 });
    program.output = 17;
    program
}

fn inputs(token_as_raw: bool) -> Vec<Input> {
    (0..4)
        .map(|r| {
            let t = (r * 2 + 1) % TOKENS;
            let first = if token_as_raw {
                SlotValue::Raw(Array1::from_shape_fn(TOKENS, |i| f64::from(u8::from(i == t))))
            } else {
                SlotValue::Token(t as u32)
            };
            vec![first, SlotValue::Raw(Array1::from_shape_fn(RAW, |i| ((r * 5 + i) as f64 * 0.9).cos()))]
        })
        .collect()
}

fn kl_at(model: &OperatorProgram, program: &OperatorProgram, family: &FamilyInputs) -> Vec<f64> {
    kl_gradients(model, program, family, 1).expect("gradients").0
}

#[test]
fn input_gradients_match_central_differences_through_every_node_kind() {
    let (model, program) = (every_kind(0.0, true), every_kind(0.05, true));
    let points = inputs(true);
    let family = family_of(&points).expect("family");
    let (kl, gradients) = kl_gradients(&model, &program, &family, 1).expect("gradients");
    assert!(kl.iter().all(|v| *v > 0.0));
    let h = 1e-6;
    for slot in 0..2 {
        let gradient = gradients[slot].as_ref().expect("both slots are read");
        for (row, point) in points.iter().enumerate() {
            let SlotValue::Raw(x) = &point[slot] else { unreachable!() };
            for i in 0..x.len() {
                let shifted = |delta: f64| {
                    let mut moved = point.clone();
                    let mut y = x.clone();
                    y[i] += delta;
                    moved[slot] = SlotValue::Raw(y);
                    kl_at(&model, &program, &family_of(&[moved]).expect("family"))[0]
                };
                let numeric = (shifted(h) - shifted(-h)) / (2.0 * h);
                let analytic = gradient[[row, i]];
                assert!(
                    (numeric - analytic).abs() <= 1e-6 * (1.0 + analytic.abs()),
                    "slot {slot} row {row} coordinate {i}: {analytic} against {numeric}"
                );
            }
        }
    }
}

#[test]
fn a_token_slots_gradient_is_the_one_hot_gradient() {
    let (tokens, relaxed) = (inputs(false), inputs(true));
    let token_grad = kl_gradients(&every_kind(0.0, false), &every_kind(0.05, false), &family_of(&tokens).expect("family"), 1)
        .expect("gradients")
        .1;
    let raw_grad = kl_gradients(&every_kind(0.0, true), &every_kind(0.05, true), &family_of(&relaxed).expect("family"), 1)
        .expect("gradients")
        .1;
    let (a, b) = (token_grad[0].as_ref().expect("read"), raw_grad[0].as_ref().expect("read"));
    assert_eq!(a.dim(), b.dim());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() <= 1e-12 * (1.0 + y.abs()), "{x} against {y}");
    }
}

fn token_domain() -> InputDomain {
    InputDomain {
        slots: vec![
            SlotDomain::Tokens((0..TOKENS as u32).collect()),
            SlotDomain::Box { lower: Array1::from_elem(RAW, -1.0), upper: Array1::from_elem(RAW, 1.0) },
        ],
    }
}

#[test]
fn every_accepted_step_raises_the_certified_kl_and_the_endpoint_has_no_better_token() {
    let (model, program) = (every_kind(0.0, false), every_kind(0.3, false));
    let starts = inputs(false);
    let domain = token_domain();
    let ascents = ascend(&model, &program, &domain, &starts, 1).expect("ascends");
    for ascent in &ascents {
        assert!(ascent.path.windows(2).all(|w| w[1] > w[0]), "path {:?}", ascent.path);
        assert!(ascent.kl.lower <= ascent.kl.value && ascent.kl.value <= ascent.kl.upper);
        assert!(ascent.kl.value >= ascent.start_kl.value);
        assert!(matches!(ascent.stop, AscentStop::Stationary | AscentStop::NoCertifiedIncrease));
        let SlotValue::Raw(x) = &ascent.input[1] else { panic!("raw slot") };
        assert!(x.iter().all(|v| (-1.0..=1.0).contains(v)));
    }
}

#[test]
fn on_an_enumerable_domain_every_ascent_stays_below_the_exhaustive_supremum() {
    let (model, program) = (every_kind(0.0, false), every_kind(0.3, false));
    let domain = InputDomain {
        slots: vec![
            SlotDomain::Tokens((0..TOKENS as u32).collect()),
            SlotDomain::Corners { lower: Array1::zeros(RAW), upper: Array1::ones(RAW) },
        ],
    };
    assert_eq!(domain.cardinality(), Some((TOKENS << RAW) as u64));
    let (worst, worst_kl, sup_upper) = exhaustive_worst(&model, &program, &domain, 1, 7).expect("enumerates");
    assert!(worst_kl.lower <= worst_kl.value && worst_kl.value <= sup_upper);
    let every: Vec<Input> = (0..TOKENS as u32)
        .flat_map(|t| {
            (0..1u32 << RAW).map(move |bits| {
                vec![SlotValue::Token(t), SlotValue::Raw(Array1::from_shape_fn(RAW, |i| f64::from(bits >> i & 1)))]
            })
        })
        .collect();
    let ascents = ascend(&model, &program, &domain, &every, 1).expect("ascends");
    for ascent in &ascents {
        assert!(ascent.kl.lower <= sup_upper);
        assert!(domain_holds(&domain, &ascent.input));
    }
    // Started everywhere, the ladder's first rung is the supremum, attained at the worst input.
    let first = ladder(&ascents, &[0])[0].1;
    assert!((first - worst_kl.value).abs() <= 1e-12 * worst_kl.value);
    let at_worst = ascents.iter().find(|a| every[a.start] == worst).expect("the worst input is a start");
    assert!((at_worst.start_kl.value - first).abs() <= 1e-12 * first);
}

fn domain_holds(domain: &InputDomain, input: &Input) -> bool {
    input.iter().zip(&domain.slots).all(|(value, slot)| match (value, slot) {
        (SlotValue::Token(t), SlotDomain::Tokens(tokens)) => tokens.contains(t),
        (SlotValue::Raw(x), SlotDomain::Corners { lower, upper }) => {
            x.iter().zip(lower).zip(upper).all(|((v, l), u)| v == l || v == u)
        }
        _ => false,
    })
}

#[test]
fn refinement_ends_with_no_ascent_above_the_familys_worst_row() {
    let model = every_kind(0.0, false);
    let domain = InputDomain {
        slots: vec![
            SlotDomain::Tokens((0..TOKENS as u32).collect()),
            SlotDomain::Corners { lower: Array1::zeros(RAW), upper: Array1::ones(RAW) },
        ],
    };
    let start = vec![vec![SlotValue::Token(0), SlotValue::Raw(Array1::zeros(RAW))]];
    let contract = Contract {
        declarations: model.declarations.clone(),
        family: family_of(&start).expect("family"),
        kind: FamilyKind::Complete { description: "the start and every counterexample".to_string() },
        observations: 64,
        readouts: 1,
        readout_slots: None,
    };
    let library: Vec<Box<dyn Primitive>> = vec![Box::new(DropBlocks), Box::new(Coarsen)];
    let budget = Budget { screenings: 100_000, certifications: 1_000, refit: None };
    let refinement = decompose_refined(&model, &contract, &library, &budget, &domain, &[]).expect("refines");
    let last = refinement.rounds.last().expect("a round");
    assert_eq!(last.counterexamples, 0);
    assert_eq!(last.rows, 1 + refinement.added);
    assert!(refinement.added as u64 <= domain.cardinality().expect("enumerable"));
    assert!(refinement.ascents.iter().all(|a| a.kl.lower <= last.data_worst_upper));
    // Every round but the last added what it found; the rounds' families grow strictly.
    assert!(refinement.rounds.windows(2).all(|w| w[1].rows == w[0].rows + w[0].counterexamples));
}

/// A rule applied to both raw slots, the halves concatenated, scaled by a gain and mapped to logits.
fn with_rule(seed: f64) -> OperatorProgram {
    let declarations =
        Declarations { parameters: 1, domains: vec![], slots: vec![Slot::Raw { width: RAW }, Slot::Raw { width: RAW }] };
    let native = Interface::native(RAW).expect("interface");
    let units = Interface::uniform(2, 1, LabelKind::Unit, 0).expect("interface");
    let rule = Rule {
        name: "unit".to_string(),
        inputs: vec![native.clone()],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Pointwise { input: 1, laws: vec![Law::Gelu, Law::Silu] },
        ],
        output: 2,
    };
    let mut program = OperatorProgram {
        declarations,
        bases: vec![],
        operators: vec![Arc::new(dense("W", &units, &native, table(2, RAW, seed)))],
        rules: vec![rule],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Raw { slot: 1 },
            Node::Call { rule: 0, arguments: vec![0] },
            Node::Call { rule: 0, arguments: vec![1] },
            Node::Concat { parts: vec![2, 3] },
            Node::Gain { input: 4, coefficient: Coefficient::Product(vec![Coefficient::Parameter(0), Coefficient::Number(0.5)]) },
        ],
        output: 5,
    };
    let joined = program.node_interface(5).expect("interface");
    let classes = Interface::uniform(CLASSES, 1, LabelKind::Token, 0).expect("interface");
    program.operators.push(Arc::new(dense("U", &classes, &joined, table(CLASSES, joined.width(), seed + 3.0))));
    program.nodes.push(Node::Affine { terms: vec![(5, 1)], bias: None });
    program.output = 6;
    program
}

#[test]
fn rules_inline_to_the_same_function_and_their_gradients_match_differences() {
    let (model, program) = (with_rule(0.0), with_rule(0.2));
    let points: Vec<Input> = (0..3)
        .map(|r| {
            (0..2)
                .map(|slot| SlotValue::Raw(Array1::from_shape_fn(RAW, |i| ((r * 3 + slot * 5 + i) as f64 * 0.7).cos())))
                .collect()
        })
        .collect();
    let family = family_of(&points).expect("family");
    let flat = inlined(&program).expect("inlines");
    assert!(flat.rules.is_empty());
    assert_eq!(
        flat.execute(&family, false).expect("runs").values[flat.output],
        program.execute(&family, false).expect("runs").values[program.output]
    );
    let (_, gradients) = kl_gradients(&model, &program, &family, 1).expect("gradients");
    let h = 1e-6;
    for slot in 0..2 {
        let gradient = gradients[slot].as_ref().expect("read");
        for (row, point) in points.iter().enumerate() {
            let SlotValue::Raw(x) = &point[slot] else { unreachable!() };
            for i in 0..RAW {
                let shifted = |delta: f64| {
                    let mut moved = point.clone();
                    let mut y = x.clone();
                    y[i] += delta;
                    moved[slot] = SlotValue::Raw(y);
                    kl_at(&model, &program, &family_of(&[moved]).expect("family"))[0]
                };
                let numeric = (shifted(h) - shifted(-h)) / (2.0 * h);
                let analytic = gradient[[row, i]];
                assert!((numeric - analytic).abs() <= 1e-6 * (1.0 + analytic.abs()), "{analytic} against {numeric}");
            }
        }
    }
}

/// Counterexamples found by the ascent are chosen for being hard, so they join the family as
/// challenge rows: fitted and scored, never counted as draws of the sampled population.
#[test]
fn counterexamples_join_as_challenge_rows_never_as_draws() {
    let model = every_kind(0.0, false);
    let domain = InputDomain {
        slots: vec![
            SlotDomain::Tokens((0..TOKENS as u32).collect()),
            SlotDomain::Corners { lower: Array1::zeros(RAW), upper: Array1::ones(RAW) },
        ],
    };
    let drawn = vec![vec![SlotValue::Token(0), SlotValue::Raw(Array1::zeros(RAW))]];
    let contract = Contract {
        declarations: model.declarations.clone(),
        family: family_of(&drawn).expect("family"),
        kind: FamilyKind::Sample { population: "one drawn input".to_string(), confidence: 0.95, units: vec![0] },
        observations: 64,
        readouts: 1,
        readout_slots: None,
    };
    let library: Vec<Box<dyn Primitive>> = vec![Box::new(DropBlocks), Box::new(Coarsen)];
    let budget = Budget { screenings: 100_000, certifications: 1_000, refit: None };
    let refinement = decompose_refined(&model, &contract, &library, &budget, &domain, &[]).expect("refines");
    assert!(refinement.added > 0, "the toy must produce counterexamples for this test to say anything");
    let population = refinement.decomposition.score.population.as_ref().expect("a sampled family");
    assert_eq!(population.units, 1);
    // Directly: two drawn rows and two challenge rows count two units.
    let inputs: Vec<Input> = (0..4u32).map(|t| vec![SlotValue::Token(t), SlotValue::Raw(Array1::zeros(RAW))]).collect();
    let mixed = Contract {
        family: family_of(&inputs).expect("family"),
        kind: FamilyKind::Sample { population: "two drawn inputs".to_string(), confidence: 0.95, units: vec![0, 1] },
        ..contract
    };
    let reference = mixed.logits(&model).expect("reference");
    let score = mixed.score(&every_kind(0.3, false), &reference).expect("score");
    assert_eq!(score.population.as_ref().expect("sampled").units, 2);
    assert_eq!(score.evaluation.argmax_agrees.len(), 4);
}
