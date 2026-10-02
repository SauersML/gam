#![cfg(test)]
//! The masked program's derivatives against finite differences of its own KL.

use super::masked::{Library, Masked, forward, gradients, sites, split, step_pieces};
use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance, Slot,
    SlotValues,
};
use super::precision::DeclaredPrecision;
use ndarray::{Array1, Array2};
use std::sync::Arc;

fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

const P: usize = 5;
const WIDTH: usize = 4;
const UNITS: usize = 6;

fn precision() -> DeclaredPrecision {
    DeclaredPrecision::new(40).expect("a precision")
}

/// Two tokens embedded, a ReLU layer, logits over five classes.
fn model() -> (OperatorProgram, FamilyInputs) {
    let tokens = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let residual = Interface::native(WIDTH).expect("interface");
    let units = Interface::uniform(UNITS, 1, LabelKind::Unit, 0).expect("interface");
    let classes = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let op = |name: &str, rows: &Interface, cols: &Interface, salt: usize| {
        let m = Array2::from_shape_fn((rows.width(), cols.width()), |(i, j)| noise(salt + 31 * i + j));
        Arc::new(Operator::dense(name, rows.clone(), cols.clone(), m, precision(), Provenance::default()).expect("dense"))
    };
    let program = OperatorProgram {
        declarations: Declarations {
            parameters: 0,
            domains: vec![Domain { size: P, cycle: None }],
            slots: vec![Slot::Token { domain: 0 }, Slot::Token { domain: 0 }],
        },
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: vec![op("E", &residual, &tokens, 1), op("W_in", &units, &residual, 100), op("W_out", &classes, &units, 200)],
        rules: Vec::new(),
        nodes: vec![
            Node::Feature { slot: 0, basis: 0 },
            Node::Feature { slot: 1, basis: 0 },
            Node::Affine { terms: vec![(0, 0), (1, 0)], bias: None },
            Node::Affine { terms: vec![(2, 1)], bias: None },
            Node::Pointwise { input: 3, laws: vec![Law::Relu; UNITS] },
            Node::Affine { terms: vec![(4, 2)], bias: None },
            Node::Readout { input: 5, basis: 0 },
        ],
        output: 6,
    };
    let pairs: Vec<(u32, u32)> = (0..P as u32).flat_map(|a| (0..P as u32).map(move |b| (a, b))).collect();
    let family = FamilyInputs {
        layout: None,
        rows: pairs.len(),
        slots: vec![SlotValues::Tokens(pairs.iter().map(|q| q.0).collect()), SlotValues::Tokens(pairs.iter().map(|q| q.1).collect())],
    };
    (program, family)
}

#[test]
fn the_masked_programs_gradients_are_its_own_kls_derivatives() {
    let (program, family) = model();
    let target = program.execute(&family, false).expect("executes").values[program.output].clone();
    let all = sites(&program);
    let site = all.iter().find(|s| s.name == "W_in").expect("the W_in site").clone();
    let pieces = 3;
    let v = Array2::from_shape_fn((pieces, WIDTH), |(i, j)| noise(700 + 7 * i + j));
    let u = Array2::from_shape_fn((pieces, UNITS), |(i, j)| noise(800 + 7 * i + j));
    let mean = Array1::from_shape_fn(WIDTH, |i| 0.1 * noise(900 + i));
    let mut masked = Masked::build(&program, vec![site], vec![Library { v: v.clone(), u: u.clone(), mean: mean.clone() }]).expect("builds");
    let masks = vec![Array2::from_shape_fn((family.rows, pieces), |(r, c)| if (r + c) % 3 == 0 { 0.0 } else { 1.0 })];
    let fam = masked.family(&family, &masks);
    let (kl, trace, cotangent) = forward(&masked, &fam, &target).expect("forward");
    let grads = gradients(&masked, &fam, &trace, &masks, cotangent).expect("gradients");
    let total = kl.sum();
    let h = 1e-6;
    for (i, j) in [(0, 0), (1, 2), (2, 3)] {
        let mut shifted = v.clone();
        shifted[[i, j]] += h;
        masked.set_library(0, Library { v: shifted, u: u.clone(), mean: mean.clone() }).expect("set");
        let (kl_v, _, _) = forward(&masked, &fam, &target).expect("forward");
        let numeric = (kl_v.sum() - total) / h;
        let analytic = grads[0].1[[i, j]];
        assert!((numeric - analytic).abs() <= 1e-4 * (1.0 + analytic.abs()), "V[{i},{j}]: {numeric} against {analytic}");
    }
    for (i, j) in [(0, 0), (1, 4), (2, 5)] {
        let mut shifted = u.clone();
        shifted[[i, j]] += h;
        masked.set_library(0, Library { v: v.clone(), u: shifted, mean: mean.clone() }).expect("set");
        let (kl_u, _, _) = forward(&masked, &fam, &target).expect("forward");
        let numeric = (kl_u.sum() - total) / h;
        let analytic = grads[0].2[[i, j]];
        assert!((numeric - analytic).abs() <= 1e-4 * (1.0 + analytic.abs()), "U[{i},{j}]: {numeric} against {analytic}");
    }
}

#[test]
fn a_step_of_the_pieces_lowers_the_masked_kl() {
    let (program, family) = model();
    let target = program.execute(&family, false).expect("executes").values[program.output].clone();
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    let pieces = 3;
    let library = Library {
        v: Array2::from_shape_fn((pieces, WIDTH), |(i, j)| noise(700 + 7 * i + j)),
        u: Array2::from_shape_fn((pieces, UNITS), |(i, j)| noise(800 + 7 * i + j)),
        mean: Array1::zeros(WIDTH),
    };
    let mut masked = Masked::build(&program, vec![site], vec![library]).expect("builds");
    let masks = vec![Array2::from_shape_fn((family.rows, pieces), |(r, c)| if (r + c) % 3 == 0 { 0.0 } else { 1.0 })];
    let fam = masked.family(&family, &masks);
    let before = forward(&masked, &fam, &target).expect("forward").0.sum();
    let mut running = super::masked::Running::default();
    assert!(step_pieces(&mut masked, &family, &target, &masks, 4, 7, &mut running).expect("steps").is_some());
    let after = forward(&masked, &fam, &target).expect("forward").0.sum();
    assert!(after < before, "{after} against {before}");
}

#[test]
fn splitting_a_library_keeps_its_sum_and_lists_both_halves_where_the_piece_was_on() {
    let pieces = 3;
    let library = Library {
        v: Array2::from_shape_fn((pieces, WIDTH), |(i, j)| noise(700 + 7 * i + j)),
        u: Array2::from_shape_fn((pieces, UNITS), |(i, j)| noise(800 + 7 * i + j)),
        mean: Array1::from_shape_fn(WIDTH, |i| 0.1 * noise(900 + i)),
    };
    let rows = 20;
    let x = Array2::from_shape_fn((rows, WIDTH), |(t, j)| noise(1000 + 5 * t + j));
    let mask = Array2::from_shape_fn((rows, pieces), |(t, c)| if c == 2 && t > 0 { 0.0 } else { 1.0 });
    let (grown, masks, origin) = split(&library, &x, &mask);
    assert_eq!(origin, vec![0, 0, 1, 1, 2]);
    // Pieces 0 and 1 are listed by every input and split; piece 2 by one input and kept whole.
    assert_eq!(grown.v.nrows(), 5);
    assert_eq!(masks.dim(), (rows, 5));
    let sum = |l: &Library| l.v.t().dot(&l.u);
    let error = (&sum(&grown) - &sum(&library)).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(error < 1e-12, "{error}");
    assert_eq!(masks.column(0), masks.column(1));
}

#[test]
fn a_diagonal_gain_is_a_column_scale_forward_and_backward() {
    let (mut program, family) = model();
    let residual = Interface::uniform(WIDTH, 1, LabelKind::Unit, 0).expect("interface");
    let gains = Array1::from_shape_fn(WIDTH, |i| 0.5 + noise(1200 + i));
    let mut present = Array2::from_elem((WIDTH, WIDTH), false);
    for i in 0..WIDTH {
        present[[i, i]] = true;
    }
    let gain = Operator::blocks("gain", residual.clone(), residual.clone(), Array2::from_diag(&gains), present, precision(), Provenance::default())
        .expect("blocks");
    assert!(gain.diagonal().is_some());
    // The embedding writes the unit-labelled residual, then the gain reads it.
    let e = program.operators[0].matrix();
    program.operators[0] = Arc::new(Operator::dense("E", residual.clone(), program.operators[0].cols.clone(), e, precision(), Provenance::default()).expect("dense"));
    let w_in = program.operators[1].matrix();
    program.operators[1] = Arc::new(Operator::dense("W_in", program.operators[1].rows.clone(), residual, w_in, precision(), Provenance::default()).expect("dense"));
    program.operators.push(Arc::new(gain));
    let gain_op = program.operators.len() - 1;
    program.nodes.insert(3, Node::Affine { terms: vec![(2, gain_op)], bias: None });
    program.nodes[4] = Node::Affine { terms: vec![(3, 1)], bias: None };
    program.nodes[5] = Node::Pointwise { input: 4, laws: vec![Law::Relu; UNITS] };
    program.nodes[6] = Node::Affine { terms: vec![(5, 2)], bias: None };
    program.nodes[7] = Node::Readout { input: 6, basis: 0 };
    program.output = 7;
    let trace = program.execute(&family, true).expect("executes");
    let rounded = program.operators[gain_op].diagonal().expect("a diagonal");
    let expected = &trace.values[2] * &rounded;
    let error = (&trace.values[3] - &expected).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(error == 0.0, "{error}");
    let cotangent = Array2::from_shape_fn(trace.values[7].dim(), |(r, c)| noise(1300 + 7 * r + c));
    let back = super::derivatives::vjp(&program, &family, &trace, cotangent.clone()).expect("reverse");
    let tangent = Array2::from_shape_fn((WIDTH, WIDTH), |(i, j)| if i == j { noise(1400 + i) } else { 0.0 });
    let tangents = [(gain_op, tangent.clone())].into_iter().collect();
    let forward = super::derivatives::jvp(&program, &family, &trace, &tangents).expect("forward");
    let left: f64 = cotangent.iter().zip(forward.iter()).map(|(a, b)| a * b).sum();
    let moved = trace.values[2].dot(&tangent.t());
    let right: f64 = back[3].as_ref().expect("a cotangent").iter().zip(moved.iter()).map(|(a, b)| a * b).sum();
    assert!((left - right).abs() <= 1e-9 * left.abs().max(1.0), "{left} against {right}");
}

#[test]
fn a_set_that_carries_over_from_the_previous_input_is_cheap_to_explain() {
    use super::masked::Context;
    // One site of 16 pieces; inputs 0..8 form one sequence whose set {2, 5, 11} never changes.
    let rows = 8;
    let mask = Array2::from_shape_fn((rows, 16), |(_, c)| if [2, 5, 11].contains(&c) { 1.0 } else { 0.0 });
    let previous: Vec<Option<usize>> = (0..rows).map(|r| r.checked_sub(1)).collect();
    let mut context = Context::new(&[16]);
    context.absorb(std::slice::from_ref(&mask), &previous);
    let coder = context.coder(previous);
    let bits = coder.bits(std::slice::from_ref(&mask));
    // The first input lists its set; every later one only confirms that three pieces stay on.
    assert!(bits[1] < bits[0], "{bits:?}");
    assert!(bits[1] < 1.0, "{bits:?}");
}

#[test]
fn pieces_grown_from_what_selection_leaves_out_recover_its_kl() {
    use super::masked::{Running, dropped_atoms, with_pieces};
    let (program, family) = model();
    let target = program.execute(&family, false).expect("executes").values[program.output].clone();
    let site = sites(&program).into_iter().find(|s| s.name == "W_in").expect("the W_in site");
    let pieces = 3;
    // Three pieces: the site map's leading three singular directions; the rest of the map is in
    // no piece.
    let w = super::masked::matrix(&program, &site).expect("matrix");
    let decomposed = super::dense::svd(w.view(), false).expect("svd");
    let mut v = Array2::<f64>::zeros((pieces, WIDTH));
    let mut u = Array2::<f64>::zeros((pieces, UNITS));
    for c in 0..pieces.min(decomposed.singular_values.len()) {
        let s = decomposed.singular_values[c].sqrt();
        v.row_mut(c).assign(&(&decomposed.vt.row(c) * s));
        u.row_mut(c).assign(&(&decomposed.u.column(c) * s));
    }
    let library = Library { v, u, mean: Array1::zeros(WIDTH) };
    let mut masked = Masked::build(&program, vec![site.clone()], vec![library]).expect("builds");
    // Piece 2 is dropped everywhere.
    let masks = vec![Array2::from_shape_fn((family.rows, pieces), |(_, c)| if c == 2 { 0.0 } else { 1.0 })];
    let mut running = Running::default();
    step_pieces(&mut masked, &family, &target, &masks, 4, 7, &mut running).expect("steps");
    let fam = masked.family(&family, &masks);
    let (before, trace, _) = forward(&masked, &fam, &target).expect("forward");
    let (v, u) = dropped_atoms(&masked, 0, &trace, &masks[0], &running, 1000.0, 0.0).expect("atoms");
    assert!(v.nrows() >= 1, "no atom");
    let grown = with_pieces(&masked.libraries[0], &v, &u).expect("grown");
    let added = v.nrows();
    let bigger = Masked::build(&program, vec![site], vec![grown]).expect("builds");
    let on = vec![Array2::from_shape_fn((family.rows, pieces + added), |(_, c)| if c == 2 { 0.0 } else { 1.0 })];
    let (after, _, _) = forward(&bigger, &bigger.family(&family, &on), &target).expect("forward");
    assert!(after.sum() < before.sum(), "{} against {}", after.sum(), before.sum());
}
