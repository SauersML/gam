#![cfg(test)]
//! Equality saturation on planted exact equivalences: each law's shorter form is found, the
//! extraction is the shortest decoded message and executes the same function within the bands,
//! gains on declared parameters stay symbolic, and a resource bound is reported as not saturated.

use std::sync::Arc;
use crate::egraph::{SaturationStop, canonical_units, normalize, saturate};
use crate::operator_program::{
    Coefficient, Declarations, FamilyInputs, Interface, Law, Node, Operator, OperatorProgram, Provenance, Scale, Slot, SlotValues,
};
use crate::precision::DeclaredPrecision;
use crate::test_support::planted_toys::dyadic;
use crate::test_support::test_governor;
use gam_runtime::resource::MemoryGovernor;
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

const WIDTH: usize = 4;

fn lattice(bits: i32) -> DeclaredPrecision {
    DeclaredPrecision::new(bits).expect("a lattice within the exponent range")
}

fn dense(name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>) -> Operator {
    Operator::dense(name, rows.clone(), cols.clone(), values, lattice(12), Provenance::native(name)).expect("a dense operator")
}

fn native(width: usize) -> Interface {
    Interface::native(width).expect("a native interface")
}

fn raw_declarations(widths: &[usize]) -> Declarations {
    Declarations { domains: Vec::new(), slots: widths.iter().map(|&width| Slot::Raw { width }).collect(), parameters: 0 }
}

fn raw_inputs(widths: &[usize], rows: usize, seed: u64) -> FamilyInputs {
    let mut rng = StdRng::seed_from_u64(seed);
    FamilyInputs { rows, slots: widths.iter().map(|&w| SlotValues::Raw(dyadic(&mut rng, rows, w, 8, 4.0))).collect(), layout: None }
}

/// The two programs' outputs agree within the sum of their execution bands on every entry.
fn assert_same_function(left: &OperatorProgram, right: &OperatorProgram, inputs: &FamilyInputs) {
    let a = left.execute(inputs, true).expect("left executes");
    let b = right.execute(inputs, true).expect("right executes");
    let (a, b) = (a.banded(left.output), b.banded(right.output));
    assert_eq!(a.values.dim(), b.values.dim());
    for ((index, x), y) in a.values.indexed_iter().zip(b.values.iter()) {
        let band = (a.bands[index] + b.bands[index]).next_up();
        assert!((x - y).abs() <= band, "entry {index:?}: {x} against {y}, band {band}");
    }
}

fn square(seed: u64) -> Array2<f64> {
    dyadic(&mut StdRng::seed_from_u64(seed), WIDTH, WIDTH, 6, 2.0)
}

#[test]
fn composed_affine_maps_extract_as_their_product() {
    let x = native(WIDTH);
    let program = OperatorProgram {
        declarations: raw_declarations(&[WIDTH]),
        bases: Vec::new(),
        rules: Vec::new(),
        operators: [dense("A", &x, &x, square(1)), dense("B", &x, &x, square(2))].into_iter().map(Arc::new).collect(),
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1)], bias: None },
        ],
        output: 2,
    };
    let normalized = normalize(&program, test_governor()).expect("normalizes");
    assert!(normalized.saturation.report.saturated(), "{:?}", normalized.saturation.report);
    assert!(normalized.bits < normalized.native_bits, "{} against {}", normalized.bits, normalized.native_bits);
    assert_eq!(normalized.program.operators.len(), 1);
    assert_eq!(normalized.bits, *normalized.extraction.candidates.iter().min().expect("a candidate"));
    let product = square(2).dot(&square(1));
    assert_eq!(normalized.program.operators[0].matrix(), product, "the dyadic product is exact");
    assert_same_function(&program, &normalized.program, &raw_inputs(&[WIDTH], 16, 3));
}

#[test]
fn terms_reading_one_input_fuse_and_equal_copies_merge() {
    let x = native(WIDTH);
    let a = square(4);
    let program = OperatorProgram {
        declarations: raw_declarations(&[WIDTH]),
        bases: Vec::new(),
        rules: Vec::new(),
        operators: [dense("A", &x, &x, a.clone()), dense("A copy", &x, &x, a.clone()), dense("B", &x, &x, square(5))].into_iter().map(Arc::new).collect(),
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(0, 1)], bias: None },
            Node::Affine { terms: vec![(1, 2), (2, 2), (0, 0)], bias: None },
        ],
        output: 3,
    };
    let normalized = normalize(&program, test_governor()).expect("normalizes");
    assert!(normalized.saturation.report.saturated());
    assert!(normalized.bits < normalized.native_bits);
    assert_eq!(normalized.program.operators.len(), 1, "B A x + B A x + A x is one operator");
    assert_eq!(normalized.program.operators[0].matrix(), square(5).dot(&a) * 2.0 + &a);
    assert_same_function(&program, &normalized.program, &raw_inputs(&[WIDTH], 16, 6));
}

#[test]
fn a_bilinear_score_with_a_constant_side_is_affine() {
    let x = native(WIDTH);
    let constant = dyadic(&mut StdRng::seed_from_u64(7), WIDTH, 1, 6, 2.0);
    let program = OperatorProgram {
        declarations: raw_declarations(&[WIDTH]),
        bases: Vec::new(),
        rules: Vec::new(),
        operators: [dense("q", &x, &Interface::constant(), constant.clone()), dense("K", &x, &x, square(8))].into_iter().map(Arc::new).collect(),
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Constant { operator: 0 },
            Node::Affine { terms: vec![(0, 1)], bias: None },
            Node::Bilinear { left: 1, right: 2, scale: Scale::InverseSqrt(4) },
        ],
        output: 3,
    };
    let normalized = normalize(&program, test_governor()).expect("normalizes");
    assert!(normalized.saturation.report.saturated());
    assert!(normalized.bits < normalized.native_bits);
    assert_eq!(normalized.program.operators.len(), 1);
    let expected = constant.t().dot(&square(8)) * 0.5;
    assert_eq!(normalized.program.operators[0].matrix(), expected);
    assert_same_function(&program, &normalized.program, &raw_inputs(&[WIDTH], 16, 9));
}

#[test]
fn an_affine_map_of_a_routed_mix_moves_through_it() {
    let x = native(WIDTH);
    let bias = |seed| dyadic(&mut StdRng::seed_from_u64(seed), WIDTH, 1, 6, 2.0);
    let program = OperatorProgram {
        declarations: raw_declarations(&[WIDTH, WIDTH, 2]),
        bases: Vec::new(),
        rules: Vec::new(),
        operators: [
            dense("B", &x, &x, square(10)),
            dense("b1", &x, &Interface::constant(), bias(11)),
            dense("b2", &x, &Interface::constant(), bias(12)),
            dense("A", &x, &x, square(13)),
        ].into_iter().map(Arc::new).collect(),
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Raw { slot: 1 },
            Node::Raw { slot: 2 },
            Node::Affine { terms: vec![(0, 0)], bias: Some(1) },
            Node::Affine { terms: vec![(1, 0)], bias: Some(2) },
            Node::Mix { weights: 2, payloads: vec![(0, 3), (1, 4)] },
            Node::Affine { terms: vec![(5, 3)], bias: None },
        ],
        output: 6,
    };
    let normalized = normalize(&program, test_governor()).expect("normalizes");
    assert!(normalized.saturation.report.saturated());
    assert!(normalized.bits <= normalized.native_bits);
    assert!(normalized.extraction.candidates.iter().all(|bits| *bits >= normalized.extraction.bits));
    assert_same_function(&program, &normalized.extraction.program, &raw_inputs(&[WIDTH, WIDTH, 2], 16, 14));
}

/// `m0·x − m1·x` with the gains on declared parameters is `(m0 − m1)·x`: at the native setting
/// `m = 1` it computes zero, but the normalized program still reads both parameters and `x`, and at
/// every other setting it computes what the original computes.
#[test]
fn gains_stay_symbolic_through_cancellation() {
    let x = native(3);
    let identity = Operator::identity("I", x.clone());
    let negated = Operator::dense("-I", x.clone(), x.clone(), -Array2::<f64>::eye(3), lattice(0), Provenance::native("-I"))
        .expect("the negated identity");
    let program = OperatorProgram {
        declarations: Declarations { parameters: 2, ..raw_declarations(&[3]) },
        bases: Vec::new(),
        rules: Vec::new(),
        operators: [identity, negated].into_iter().map(Arc::new).collect(),
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Gain { input: 0, coefficient: Coefficient::Parameter(0) },
            Node::Gain { input: 0, coefficient: Coefficient::Parameter(1) },
            Node::Affine { terms: vec![(1, 0), (2, 1)], bias: None },
        ],
        output: 3,
    };
    let normalized = normalize(&program, test_governor()).expect("normalizes");
    let saturation = &normalized.saturation;
    assert!(saturation.report.saturated());
    let choices = &normalized.extraction.choices;
    let term = saturation.render(choices, saturation.root);
    assert_eq!(saturation.parameters_read(choices, saturation.root).into_iter().collect::<Vec<_>>(), vec![0, 1], "{term}");
    assert!(term.contains("x0"), "the input survives: {term}");
    assert!(term.starts_with('(') && term.contains("m0") && term.contains("m1") && term.ends_with("·x0"), "{term}");
    let lowered = &normalized.extraction.program;
    assert!(lowered.nodes.iter().any(|node| matches!(node, Node::Gain { .. })), "the gain is a node of the program");
    let inputs = raw_inputs(&[3], 8, 15);
    for setting in [[1.0, 1.0], [3.0, 1.0], [0.5, -2.0]] {
        let a = program.execute_at(&inputs, true, &setting).expect("the original executes").banded(program.output);
        let b = lowered.execute_at(&inputs, true, &setting).expect("the normalized executes").banded(lowered.output);
        for ((index, u), v) in a.values.indexed_iter().zip(b.values.iter()) {
            let band = (a.bands[index] + b.bands[index]).next_up();
            assert!((u - v).abs() <= band, "setting {setting:?}, entry {index:?}: {u} against {v}");
        }
    }
}

#[test]
fn a_resource_bound_is_reported_as_not_saturated() {
    let x = native(WIDTH);
    let program = OperatorProgram {
        declarations: raw_declarations(&[WIDTH]),
        bases: Vec::new(),
        rules: Vec::new(),
        operators: [dense("A", &x, &x, square(16)), dense("B", &x, &x, square(17)), dense("C", &x, &x, square(18))].into_iter().map(Arc::new).collect(),
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1), (0, 2)], bias: None },
            Node::Affine { terms: vec![(2, 2), (1, 0)], bias: None },
        ],
        output: 3,
    };
    let governor = MemoryGovernor::with_budget_bytes(1);
    let bounded = saturate(&program, &governor).expect("a bounded saturation");
    assert!(matches!(bounded.report.stop, SaturationStop::ResourceBound { .. }));
    assert!(!bounded.report.saturated());
    let extraction = bounded.extract(&HashMap::new()).expect("still extracts");
    assert_same_function(&program, &extraction.program, &raw_inputs(&[WIDTH], 8, 19));
    let full = saturate(&program, test_governor()).expect("saturates");
    assert!(full.report.saturated());
}

/// A two-layer ReLU block, its hidden units rescaled by powers of two, shuffled and one unit
/// duplicated (read row copied, write column halved), has the same canonical form.
#[test]
fn the_unit_gauge_is_canonical_under_rescaling_shuffling_and_duplication() {
    let hidden = 6;
    let mut rng = StdRng::seed_from_u64(20);
    let read = dyadic(&mut rng, hidden, WIDTH, 6, 2.0);
    let bias = dyadic(&mut rng, hidden, 1, 6, 4.0);
    let write = dyadic(&mut rng, WIDTH, hidden, 6, 2.0);
    let block = |read: Array2<f64>, bias: Array2<f64>, write: Array2<f64>| {
        let units = Interface::uniform(read.nrows(), 1, crate::operator_program::LabelKind::Unit, 0)
            .expect("unit interface");
        let x = native(WIDTH);
        OperatorProgram {
            declarations: raw_declarations(&[WIDTH]),
            bases: Vec::new(),
            rules: Vec::new(),
            operators: [
                dense("read", &units, &x, read.clone()),
                dense("bias", &units, &Interface::constant(), bias),
                dense("write", &x, &units, write),
            ].into_iter().map(Arc::new).collect(),
            nodes: vec![
                Node::Raw { slot: 0 },
                Node::Affine { terms: vec![(0, 0)], bias: Some(1) },
                Node::Pointwise { input: 1, laws: vec![Law::Relu; read.nrows()] },
                Node::Affine { terms: vec![(2, 2)], bias: None },
            ],
            output: 3,
        }
    };
    let original = block(read.clone(), bias.clone(), write.clone());
    let order = [3, 0, 5, 1, 4, 2];
    let scales = [4.0, 0.5, 1.0, 2.0, 8.0, 0.25];
    let mut read2 = Array2::zeros((hidden + 1, WIDTH));
    let mut bias2 = Array2::zeros((hidden + 1, 1));
    let mut write2 = Array2::zeros((WIDTH, hidden + 1));
    for (position, &unit) in order.iter().enumerate() {
        read2.row_mut(position).assign(&read.row(unit).mapv(|v| v * scales[unit]));
        bias2[[position, 0]] = bias[[unit, 0]] * scales[unit];
        write2.column_mut(position).assign(&write.column(unit).mapv(|v| v / scales[unit]));
    }
    let copy = 3;
    let copied = read2.row(copy).to_owned();
    read2.row_mut(hidden).assign(&copied);
    bias2[[hidden, 0]] = bias2[[copy, 0]];
    let halved = write2.column(copy).mapv(|v| v / 2.0);
    write2.column_mut(copy).assign(&halved);
    write2.column_mut(hidden).assign(&halved);
    let variant = block(read2, bias2, write2);
    assert_same_function(&original, &variant, &raw_inputs(&[WIDTH], 16, 21));
    let (left, right) = (canonical_units(&original).expect("canonical"), canonical_units(&variant).expect("canonical"));
    for (a, b) in left.operators.iter().zip(&right.operators) {
        assert_eq!(a.matrix(), b.matrix(), "operator {}", a.name);
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
    }
    assert_eq!(left.nodes, right.nodes);
    assert_same_function(&original, &left, &raw_inputs(&[WIDTH], 16, 22));
    let unit_rows = left.operators[0].matrix();
    let unit_bias = left.operators[1].matrix();
    for (row, &bias) in unit_rows.outer_iter().zip(unit_bias.column(0)) {
        let norm = row.iter().chain(std::iter::once(&bias)).map(|v| v * v).sum::<f64>().sqrt();
        let scale = if bias != 0.0 { bias.abs() } else { norm };
        assert!((1.0..2.0).contains(&scale), "a balanced unit's bias (its read row's norm without one) is in [1, 2): {scale}");
    }
    assert_eq!(unit_rows.nrows(), hidden, "the duplicate merged");
}
