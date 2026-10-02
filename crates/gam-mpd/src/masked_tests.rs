#![cfg(test)]
//! The masked program's derivatives against finite differences of its own KL.

use super::masked::{Library, Masked, forward, gradients, sites, step_pieces};
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
    assert!(step_pieces(&mut masked, &family, &target, &masks, 4, 7).expect("steps").is_some());
    let after = forward(&masked, &fam, &target).expect("forward").0.sum();
    assert!(after < before, "{after} against {before}");
}
