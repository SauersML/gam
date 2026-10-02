#![cfg(test)]
//! Quotients on planted programs: the power-of-two gauge and the single-key collapse execute
//! bit-identically, the shifts keep every readout distribution within the banded executions'
//! radii, the savings are the message's own, and non-homogeneous laws are not scaled.

use std::sync::Arc;
use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram,
    Provenance, Scale, Slot, SlotValues,
};
use super::precision::DeclaredPrecision;
use super::quotient::{Invariance, index_bits, quotient, scale_gauge, shortest_shift, single_keys};
use gam_linalg::roundoff::UNIT_ROUNDOFF;
use ndarray::{Array2, s};

const TOKENS: usize = 5;
const UNITS: usize = 6;

fn precision(bits: i32) -> DeclaredPrecision {
    DeclaredPrecision::new(bits).expect("a precision in range")
}

/// Deterministic reals of mixed sign and magnitude.
fn reals(rows: usize, cols: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| ((i * 13 + j * 7 + salt * 5) as f64 * 0.731).sin() * 1.3)
}

fn dense(name: &str, rows: &Interface, cols: &Interface, salt: usize) -> Operator {
    Operator::dense(name, rows.clone(), cols.clone(), reals(rows.width(), cols.width(), salt), precision(12), Provenance::native(name))
        .expect("dense")
}

/// Two causal positions over five tokens: embed with position biases, one attention head with a
/// shared key bias, a residual, a shared unit MLP of `law`, and logits at both positions
/// (`readouts = 2`). Position 0's softmax has one score.
fn transformer(law: Law) -> OperatorProgram {
    let declarations = Declarations {
        domains: vec![Domain { size: TOKENS, cycle: None }],
        slots: vec![Slot::Token { domain: 0 }, Slot::Token { domain: 0 }],
        parameters: 0,
    };
    let tokens = Interface::uniform(TOKENS, 1, LabelKind::Token, 0).expect("interface");
    let model = Interface::native(4).expect("interface");
    let head = Interface::native(3).expect("interface");
    let units = Interface::uniform(UNITS, 1, LabelKind::Unit, 0).expect("interface");
    let classes = Interface::uniform(TOKENS, 1, LabelKind::Token, 0).expect("interface");
    let constant = Interface::constant();
    let operators = vec![
        dense("E", &model, &tokens, 0),
        dense("pos0", &model, &constant, 1),
        dense("pos1", &model, &constant, 2),
        dense("WQ", &head, &model, 3),
        dense("WK", &head, &model, 4),
        dense("bK", &head, &constant, 5),
        dense("WV", &head, &model, 6),
        dense("WO", &model, &head, 7),
        Operator::identity("I", model.clone()),
        dense("Win", &units, &model, 9),
        dense("bin", &units, &constant, 10),
        dense("Wout", &model, &units, 11),
        dense("WU", &classes, &model, 12),
        dense("bU", &classes, &constant, 13),
    ];
    let score = Scale::InverseSqrt(3);
    let nodes = vec![
        Node::Feature { slot: 0, basis: 0 },
        Node::Feature { slot: 1, basis: 0 },
        Node::Affine { terms: vec![(0, 0)], bias: Some(1) },
        Node::Affine { terms: vec![(1, 0)], bias: Some(2) },
        Node::Affine { terms: vec![(2, 3)], bias: None },
        Node::Affine { terms: vec![(3, 3)], bias: None },
        Node::Affine { terms: vec![(2, 4)], bias: Some(5) },
        Node::Affine { terms: vec![(3, 4)], bias: Some(5) },
        Node::Affine { terms: vec![(2, 6)], bias: None },
        Node::Affine { terms: vec![(3, 6)], bias: None },
        Node::Bilinear { left: 4, right: 6, scale: score },
        Node::Softmax { scores: vec![10] },
        Node::Mix { weights: 11, payloads: vec![(0, 8)] },
        Node::Bilinear { left: 5, right: 6, scale: score },
        Node::Bilinear { left: 5, right: 7, scale: score },
        Node::Softmax { scores: vec![13, 14] },
        Node::Mix { weights: 15, payloads: vec![(0, 8), (1, 9)] },
        Node::Affine { terms: vec![(2, 8), (12, 7)], bias: None },
        Node::Affine { terms: vec![(3, 8), (16, 7)], bias: None },
        Node::Affine { terms: vec![(17, 9)], bias: Some(10) },
        Node::Pointwise { input: 19, laws: vec![law; UNITS] },
        Node::Affine { terms: vec![(17, 8), (20, 11)], bias: None },
        Node::Affine { terms: vec![(18, 9)], bias: Some(10) },
        Node::Pointwise { input: 22, laws: vec![law; UNITS] },
        Node::Affine { terms: vec![(18, 8), (23, 11)], bias: None },
        Node::Affine { terms: vec![(21, 12)], bias: Some(13) },
        Node::Affine { terms: vec![(24, 12)], bias: Some(13) },
        Node::Concat { parts: vec![25, 26] },
    ];
    OperatorProgram {
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: operators.into_iter().map(Arc::new).collect(),
        rules: Vec::new(),
        nodes,
        output: 27,
    }
}

fn family(slots: usize) -> FamilyInputs {
    let rows = TOKENS.pow(slots as u32);
    FamilyInputs {
        rows,
        slots: (0..slots)
            .map(|k| SlotValues::Tokens((0..rows).map(|r| ((r / TOKENS.pow(k as u32)) % TOKENS) as u32).collect()))
            .collect(),
        layout: None,
    }
}

fn output(program: &OperatorProgram, inputs: &FamilyInputs) -> Array2<f64> {
    let trace = program.execute(inputs, false).expect("execute");
    trace.values[program.output].clone()
}

/// Each block's log-softmax of the two banded outputs agrees within `2(R₁ + R₂)` (log-softmax is
/// 2-Lipschitz in the sup norm) plus the rounding of the two log-softmax evaluations.
fn assert_distributions_within_bands(a: &OperatorProgram, b: &OperatorProgram, inputs: &FamilyInputs, readouts: usize) {
    let (ta, tb) = (a.execute(inputs, true).expect("banded"), b.execute(inputs, true).expect("banded"));
    let (va, vb) = (&ta.values[a.output], &tb.values[b.output]);
    let (ra, rb) = (ta.banded(a.output).bands, tb.banded(b.output).bands);
    let width = va.ncols() / readouts;
    let lsm = |row: ndarray::ArrayView1<f64>| {
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
        row.mapv(|v| v - lse)
    };
    for r in 0..va.nrows() {
        for block in 0..readouts {
            let cols = s![r, block * width..(block + 1) * width];
            let (la, lb) = (lsm(va.slice(cols)), lsm(vb.slice(cols)));
            let radius = ra.slice(cols).iter().copied().fold(0.0, f64::max) + rb.slice(cols).iter().copied().fold(0.0, f64::max);
            let scale = la.iter().chain(lb.iter()).chain(va.slice(cols).iter()).chain(vb.slice(cols).iter()).fold(0.0_f64, |m, v| m.max(v.abs()));
            let slack = 2.0 * radius + 16.0 * width as f64 * UNIT_ROUNDOFF * scale;
            for (x, y) in la.iter().zip(lb.iter()) {
                assert!((x - y).abs() <= slack, "row {r} block {block}: {x} vs {y}, slack {slack}");
            }
        }
    }
}

#[test]
fn single_key_softmax_collapses_bit_identically() {
    let program = transformer(Law::Relu);
    let collapsed = single_keys(&program).expect("position 0 has one key");
    assert!(collapsed.nodes.len() < program.nodes.len());
    assert!(collapsed.nodes.iter().all(|n| !matches!(n, Node::Softmax { scores } if scores.len() == 1)));
    let inputs = family(2);
    assert_eq!(output(&program, &inputs), output(&collapsed, &inputs));
}

/// Unit `i`'s read rows scaled by `2^{k_i}` and its write column by `2^{−k_i}`: an exact restatement.
fn restated_units(program: &OperatorProgram, exponents: &[i32]) -> OperatorProgram {
    let mut out = program.clone();
    let names = ["Win", "bin", "Wout"];
    let find = |name: &str| out.operators.iter().position(|o| o.name == name).expect("operator");
    let (win, bin, wout) = (find(names[0]), find(names[1]), find(names[2]));
    let fine = precision(12 + exponents.iter().copied().max().unwrap_or(0));
    for &(op, row) in &[(win, true), (bin, true), (wout, false)] {
        let operator = Arc::make_mut(&mut out.operators[op]);
        let OperatorBody::Dense { values, precision: p, .. } = &mut operator.body else { panic!("dense") };
        for (i, &k) in exponents.iter().enumerate() {
            let factor = 2f64.powi(if row { k } else { -k });
            if row {
                values.row_mut(i).mapv_inplace(|v| v * factor);
            } else {
                values.column_mut(i).mapv_inplace(|v| v * factor);
            }
        }
        if !row {
            *p = fine;
        }
    }
    out
}

#[test]
fn power_of_two_gauge_is_bit_identical_and_undoes_a_restatement() {
    let program = transformer(Law::Relu);
    let restated = restated_units(&program, &[5, 0, 3, 7, 0, 2]);
    let inputs = family(2);
    assert_eq!(output(&program, &inputs), output(&restated, &inputs), "the restatement is exact");
    let (moved, generators) = scale_gauge(&restated).expect("gauge");
    let moved = moved.expect("a restated program has a shorter representative");
    assert_eq!(output(&restated, &inputs), output(&moved, &inputs), "the gauge executes bit-identically");
    assert!(moved.code_bits().expect("bits") < restated.code_bits().expect("bits"));
    // Residual 4, query/key 3, value/output 3, units 6.
    assert_eq!(generators, 16);
}

#[test]
fn non_homogeneous_units_are_not_scaled() {
    let (_, relu) = scale_gauge(&transformer(Law::Relu)).expect("gauge");
    let (_, gelu) = scale_gauge(&transformer(Law::Gelu)).expect("gauge");
    assert_eq!(relu - gelu, UNITS);
}

#[test]
fn quotient_saves_exact_message_bits_and_keeps_every_distribution() {
    for law in [Law::Relu, Law::Gelu] {
        let program = transformer(law);
        let q = quotient(&program, 2).expect("quotient");
        assert_eq!(q.bits_before, program.code_bits().expect("bits"));
        assert_eq!(q.bits_after, q.program.code_bits().expect("bits"));
        assert!(q.bits_after < q.bits_before);
        assert_eq!(q.saved.values().sum::<u64>(), q.bits_before - q.bits_after);
        assert!(q.saved.contains_key(&Invariance::SoftmaxShift), "the key bias is removed: {:?}", q.saved);
        assert!(q.program.operators.iter().all(|o| o.name != "bK"));
        let message = q.program.encode().expect("encode");
        assert_eq!(message.len_bits(), q.bits_after);
        let decoded = OperatorProgram::decode(&message, &q.program.declarations).expect("decode");
        assert_eq!(decoded.code_bits().expect("bits"), q.bits_after);
        for (a, b) in decoded.operators.iter().zip(&q.program.operators) {
            assert_eq!(a.matrix(), b.matrix(), "operator {} decodes exactly", b.name);
        }
        assert!(q.saved.contains_key(&Invariance::Permutation), "interchangeable coordinates are sorted: {:?}", q.saved);
        assert_distributions_within_bands(&program, &q.program, &family(2), 2);
        let again = quotient(&q.program, 2).expect("quotient");
        assert_eq!(again.bits_after, again.bits_before, "the representative is a fixed point");
    }
}

#[test]
fn character_readout_drops_its_constant_row() {
    let cycle = Some((0..TOKENS as u32).map(Some).collect::<Vec<_>>());
    let declarations = Declarations {
        domains: vec![Domain { size: TOKENS, cycle: None }, Domain { size: TOKENS, cycle: cycle.clone() }],
        slots: vec![Slot::Token { domain: 0 }],
        parameters: 0,
    };
    let characters = Basis::Characters { domain: 1, positions: (0..TOKENS as u32).map(Some).collect(), declared: true };
    let tokens = Interface::uniform(TOKENS, 1, LabelKind::Token, 0).expect("interface");
    let model = Interface::native(4).expect("interface");
    let basis = characters.interface(&declarations).expect("interface");
    let program = OperatorProgram {
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }, characters],
        operators: vec![Arc::new(dense("E", &model, &tokens, 0)), Arc::new(dense("U", &basis, &model, 1))],
        rules: Vec::new(),
        nodes: vec![
            Node::Feature { slot: 0, basis: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1)], bias: None },
            Node::Readout { input: 2, basis: 1 },
        ],
        output: 3,
    };
    let q = quotient(&program, 1).expect("quotient");
    let u = q.program.operators.iter().find(|o| o.name == "U").expect("U");
    let OperatorBody::Dense { present, .. } = &u.body else { panic!("dense") };
    assert!(!present[[0, 0]], "the constant row is sent absent");
    assert!(q.saved.contains_key(&Invariance::LogitShift));
    assert_distributions_within_bands(&program, &q.program, &family(1), 1);
}

#[test]
fn shortest_shift_is_the_exact_minimiser() {
    let sets: Vec<Vec<i64>> = vec![
        vec![0, 16],
        vec![100, 101, 99, 130, -7],
        vec![-5, -5, -5, 3],
        vec![1 << 20, (1 << 20) + 3, (1 << 20) - 9, 17],
        (0..40).map(|i| ((i * 37) % 23) as i64 * 11 - 60).collect(),
    ];
    let cost = |set: &[i64], d: i64| set.iter().map(|n| index_bits(n - d).expect("bits")).sum::<u64>();
    for set in sets {
        let found = shortest_shift(&set).expect("shift");
        let (lo, hi) = (*set.iter().min().unwrap(), *set.iter().max().unwrap());
        let best = (lo.min(0) - 2..=hi.max(0) + 2).map(|d| cost(&set, d)).min().unwrap();
        assert_eq!(cost(&set, found), best, "{set:?}: shift {found}");
        if cost(&set, 0) == best {
            assert_eq!(found, 0, "{set:?}: zero is kept when nothing is shorter");
        }
    }
}

#[test]
fn ordered_rows_are_sent_without_their_order() {
    let program = transformer(Law::Relu);
    let win = program.operators.iter().position(|o| o.name == "Win").expect("Win");
    let rows_in = |order: &[usize]| {
        let mut out = program.clone();
        let OperatorBody::Dense { values, .. } = &mut Arc::make_mut(&mut out.operators[win]).body else { panic!("dense") };
        let source = values.clone();
        for (to, &from) in order.iter().enumerate() {
            values.row_mut(to).assign(&source.row(from));
        }
        out
    };
    let OperatorBody::Dense { values, .. } = &program.operators[win].body else { panic!("dense") };
    let mut ascending: Vec<usize> = (0..UNITS).collect();
    ascending.sort_by(|&a, &b| values[[a, 0]].total_cmp(&values[[b, 0]]));
    let descending: Vec<usize> = ascending.iter().rev().copied().collect();
    let (sorted, reversed) = (rows_in(&ascending), rows_in(&descending));
    assert!(sorted.code_bits().expect("bits") < reversed.code_bits().expect("bits"));
    let message = sorted.encode().expect("encode");
    assert_eq!(message.len_bits(), sorted.code_bits().expect("bits"));
    let decoded = OperatorProgram::decode(&message, &sorted.declarations).expect("decode");
    assert_eq!(decoded.operators[win].matrix(), sorted.operators[win].matrix());
}

#[test]
fn equal_operators_are_read_as_one() {
    let mut program = transformer(Law::Relu);
    let wu = program.operators.iter().position(|o| o.name == "WU").expect("WU");
    let mut copy = Operator::clone(&program.operators[wu]);
    copy.name = "WU copy".to_string();
    program.operators.push(Arc::new(copy));
    let copy = program.operators.len() - 1;
    program.nodes[26] = Node::Affine { terms: vec![(24, copy)], bias: Some(13) };
    let q = quotient(&program, 2).expect("quotient");
    assert!(q.saved.contains_key(&Invariance::Tie), "{:?}", q.saved);
    assert!(q.program.operators.iter().all(|o| o.name != "WU copy"));
    assert_distributions_within_bands(&program, &q.program, &family(2), 2);
}
