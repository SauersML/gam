#![cfg(test)]
//! Operator programs: the message decodes to the program and has the computed length, and every
//! banded execution's radius covers the quad-double evaluation of the same program.

use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram,
    Provenance, Scale, Slot, SlotValues,
};
use super::precision::DeclaredPrecision;
use ndarray::{Array2, array};
use qd::Quad;
use std::sync::Arc;

fn precision(bits: i32) -> DeclaredPrecision {
    DeclaredPrecision::new(bits).expect("a precision in range")
}

/// Deterministic reals of mixed sign and magnitude, not dyadic.
fn reals(rows: usize, cols: usize, salt: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| ((i * 13 + j * 7 + salt * 5) as f64 * 0.731).sin() * 1.3)
}

/// A one-layer routing program over two token slots: embed, score, route, read, ReLU units,
/// write logits over a declared class domain.
fn fixture() -> OperatorProgram {
    let declarations = Declarations { parameters: 0,
        domains: vec![
            Domain { size: 7, cycle: Some((0..7).map(|t| (t < 5).then_some(t as u32)).collect()) },
            Domain { size: 5, cycle: None },
        ],
        slots: vec![Slot::Token { domain: 0 }, Slot::Token { domain: 0 }],
    };
    let p = precision(20);
    let tokens = Interface::uniform(7, 1, LabelKind::Token, 0).expect("interface");
    let model = Interface::native(4).expect("interface");
    let head = Interface::native(3).expect("interface");
    let score = Interface::native(1).expect("interface");
    let units = Interface::uniform(5, 1, LabelKind::Unit, 0).expect("interface");
    let classes = Interface::uniform(5, 1, LabelKind::Token, 0).expect("interface");
    let constant = Interface::constant();
    let dense = |name: &str, rows: &Interface, cols: &Interface, salt: usize| {
        Operator::dense(name, rows.clone(), cols.clone(), reals(rows.width(), cols.width(), salt), p, Provenance::native(name))
            .expect("a dense operator")
    };
    let operators = vec![
        dense("E", &model, &tokens, 1),
        dense("pos0", &model, &constant, 2),
        dense("pos1", &model, &constant, 3),
        dense("Q", &score, &model, 4),
        dense("V", &head, &model, 5),
        dense("W_in", &units, &head, 6),
        dense("b_in", &units, &constant, 7),
        dense("W_out", &classes, &units, 8),
        dense("W_direct", &classes, &head, 9),
    ];
    let nodes = vec![
        Node::Feature { slot: 0, basis: 0 },
        Node::Feature { slot: 1, basis: 0 },
        Node::Affine { terms: vec![(0, 0)], bias: Some(1) },
        Node::Affine { terms: vec![(1, 0)], bias: Some(2) },
        Node::Affine { terms: vec![(2, 3)], bias: None },
        Node::Bilinear { left: 2, right: 3, scale: Scale::InverseSqrt(4) },
        Node::Softmax { scores: vec![4, 5] },
        Node::Affine { terms: vec![(2, 4)], bias: None },
        Node::Affine { terms: vec![(3, 4)], bias: None },
        Node::Mix { weights: 6, payloads: vec![(0, 7), (1, 8)] },
        Node::Hadamard { left: 9, right: 7 },
        Node::Affine { terms: vec![(10, 5)], bias: Some(6) },
        Node::Pointwise { input: 11, laws: vec![Law::Relu, Law::Silu, Law::Identity, Law::Zero, Law::Relu] },
        Node::Affine { terms: vec![(12, 7), (9, 8)], bias: None },
        Node::Readout { input: 13, basis: 1 },
        Node::Outer { left: 12, right: 6 },
        Node::Concat { parts: vec![9, 6, 12] },
    ];
    OperatorProgram { rules: Vec::new(),
        declarations,
        bases: vec![Basis::Indicator { domain: 0 }, Basis::Indicator { domain: 1 }],
        operators: operators.into_iter().map(Arc::new).collect(),
        nodes,
        output: 14,
    }
}

fn family() -> FamilyInputs {
    let pairs: Vec<(u32, u32)> = (0..7).flat_map(|a| (0..7).map(move |b| (a, b))).collect();
    FamilyInputs {
        layout: None,
        rows: pairs.len(),
        slots: vec![
            SlotValues::Tokens(pairs.iter().map(|p| p.0).collect()),
            SlotValues::Tokens(pairs.iter().map(|p| p.1).collect()),
        ],
    }
}

fn same_structure(left: &OperatorProgram, right: &OperatorProgram) {
    assert_eq!(left.bases, right.bases);
    assert_eq!(left.nodes, right.nodes);
    assert_eq!(left.output, right.output);
    assert_eq!(left.operators.len(), right.operators.len());
    for (a, b) in left.operators.iter().zip(&right.operators) {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        assert_eq!(a.body, b.body);
    }
}

#[test]
fn message_decodes_to_the_program_at_its_computed_length() {
    let mut program = fixture();
    // Absent blocks and a coarse lattice must survive the round trip too.
    if let OperatorBody::Dense { values, present, precision: q } = &program.operators[5].body {
        let mut present = present.clone();
        present[[1, 0]] = false;
        present[[3, 0]] = false;
        program.operators[5] = Arc::new(Operator::blocks(
            "W_in",
            program.operators[5].rows.clone(),
            program.operators[5].cols.clone(),
            values.clone(),
            present,
            *q,
            Provenance::default(),
        )
        .expect("blocks"));
    }
    program.operators[0] = Arc::new(Operator::dense(
        "E",
        program.operators[0].rows.clone(),
        program.operators[0].cols.clone(),
        program.operators[0].matrix(),
        precision(-1),
        Provenance::default(),
    )
    .expect("coarse"));
    program.bases[0] = Basis::Characters {
        domain: 0,
        positions: vec![Some(3), Some(0), Some(4), Some(1), Some(2), None, None],
        declared: false,
    };
    // A character basis has width 1 + 2·2 + 2 = 7, the indicator's width, with different groups.
    let character_cols = program.bases[0].interface(&program.declarations).expect("interface");
    program.operators[0] = Arc::new(Operator::dense(
        "E",
        program.operators[0].rows.clone(),
        character_cols,
        program.operators[0].matrix(),
        precision(12),
        Provenance::default(),
    )
    .expect("character columns"));
    let message = program.encode().expect("encodes");
    assert_eq!(message.len_bits(), program.code_bits().expect("length"));
    let decoded = OperatorProgram::decode(&message, &program.declarations).expect("decodes");
    same_structure(&program, &decoded);
    // A declared labelling is sent as one bit and read back from the declarations.
    program.bases[0] = Basis::Characters {
        domain: 0,
        positions: program.declarations.domains[0].cycle.clone().expect("declared"),
        declared: true,
    };
    let declared = program.encode().expect("encodes");
    assert_eq!(declared.len_bits(), program.code_bits().expect("length"));
    assert!(declared.len_bits() < message.len_bits());
    same_structure(&program, &OperatorProgram::decode(&declared, &program.declarations).expect("decodes"));
}

#[test]
fn a_truncated_or_padded_message_is_refused() {
    let program = fixture();
    let message = program.encode().expect("encodes");
    let mut padded = message.clone();
    padded.push_bit(true);
    assert!(OperatorProgram::decode(&padded, &program.declarations).is_err());
    let mut truncated = super::codec::BitString::new();
    let mut reader = message.reader();
    for _ in 0..message.len_bits() - 1 {
        truncated.push_bit(reader.read_bit().expect("bit"));
    }
    assert!(OperatorProgram::decode(&truncated, &program.declarations).is_err());
}

fn q(value: f64) -> Quad {
    Quad::from_f64(value)
}

fn to_f64(value: Quad) -> f64 {
    value.0 + value.1
}

/// The fixture evaluated in quad-double at the decoded reals: every node, every row.
fn quad_evaluation(program: &OperatorProgram, inputs: &FamilyInputs) -> Vec<Vec<Vec<Quad>>> {
    let interfaces = program.interfaces().expect("valid");
    let mut values: Vec<Vec<Vec<Quad>>> = Vec::new();
    for (index, node) in program.nodes.iter().enumerate() {
        let width = interfaces[index].width();
        let mut rows = Vec::with_capacity(inputs.rows);
        for row in 0..inputs.rows {
            let mut out = vec![q(0.0); width];
            match node {
                Node::Feature { slot, .. } => {
                    let SlotValues::Tokens(tokens) = &inputs.slots[*slot] else { unreachable_input() };
                    out[tokens[row] as usize] = q(1.0);
                }
                Node::Affine { terms, bias } => {
                    for (argument, operator) in terms {
                        let a = program.operators[*operator].matrix();
                        for (o, target) in out.iter_mut().enumerate() {
                            for i in 0..a.ncols() {
                                *target += q(a[[o, i]]) * values[*argument][row][i];
                            }
                        }
                    }
                    if let Some(op) = bias {
                        let b = program.operators[*op].matrix();
                        for (o, target) in out.iter_mut().enumerate() {
                            *target += q(b[[o, 0]]);
                        }
                    }
                }
                Node::Bilinear { left, right, scale } => {
                    let mut total = q(0.0);
                    for i in 0..values[*left][row].len() {
                        total += values[*left][row][i] * values[*right][row][i];
                    }
                    out[0] = total * q(scale.value());
                }
                Node::Softmax { scores } => {
                    let exps: Vec<Quad> = scores.iter().map(|s| values[*s][row][0].exp()).collect();
                    let total = exps.iter().fold(q(0.0), |acc, e| acc + *e);
                    for (j, e) in exps.iter().enumerate() {
                        out[j] = *e / total;
                    }
                }
                Node::Mix { weights, payloads } => {
                    for &(j, payload) in payloads.iter() {
                        let payload = &payload;
                        for (c, target) in out.iter_mut().enumerate() {
                            *target += values[*weights][row][j] * values[*payload][row][c];
                        }
                    }
                }
                Node::Hadamard { left, right } => {
                    for (c, target) in out.iter_mut().enumerate() {
                        *target = values[*left][row][c] * values[*right][row][c];
                    }
                }
                Node::Pointwise { input, laws } => {
                    let interface = &interfaces[*input];
                    for (group, law) in laws.iter().enumerate() {
                        for c in interface.range(group) {
                            let x = values[*input][row][c];
                            out[c] = match law {
                                Law::Relu => {
                                    if x.gt(q(0.0)) {
                                        x
                                    } else {
                                        q(0.0)
                                    }
                                }
                                Law::Identity => x,
                                Law::Zero => q(0.0),
                                Law::Silu => x / (q(1.0) + (-x).exp()),
                                Law::Gelu | Law::GeluTanh => unreachable_input(),
                            };
                        }
                    }
                }
                Node::Readout { input, .. } => out.clone_from(&values[*input][row]),
                Node::Concat { parts } => {
                    let mut offset = 0;
                    for part in parts {
                        for value in &values[*part][row] {
                            out[offset] = *value;
                            offset += 1;
                        }
                    }
                }
                Node::Outer { left, right } => {
                    let (li, ri) = (&interfaces[*left], &interfaces[*right]);
                    let mut offset = 0;
                    for g1 in 0..li.group_count() {
                        for g2 in 0..ri.group_count() {
                            for i in li.range(g1) {
                                for j in ri.range(g2) {
                                    out[offset] = values[*left][row][i] * values[*right][row][j];
                                    offset += 1;
                                }
                            }
                        }
                    }
                }
                Node::Raw { .. }
                | Node::Constant { .. }
                | Node::Param { .. }
                | Node::Call { .. }
                | Node::Gain { .. }
                | Node::Attend { .. }
                | Node::RmsNorm { .. }
                | Node::Transposed { .. } => {
                    unreachable_input()
                }
            }
            rows.push(out);
        }
        values.push(rows);
    }
    values
}

fn unreachable_input() -> ! {
    // SAFETY: the fixture has no raw slots or constants and only token slots feed its features.
    panic!("the quad evaluator covers only the fixture's node kinds")
}

#[test]
fn every_radius_covers_the_quad_double_value() {
    let program = fixture();
    let inputs = family();
    let trace = program.execute(&inputs, true).expect("executes");
    let bands: Vec<Array2<f64>> = (0..program.nodes.len()).map(|n| trace.band(n).expect("bands were requested")).collect();
    let exact = quad_evaluation(&program, &inputs);
    let mut widest_ratio = 0.0_f64;
    for (node, rows) in exact.iter().enumerate() {
        // A gathered feature holds no columns in the trace: its one-hot rows, formed here, are
        // exact, so their radius is zero.
        let gathered = program.gathered_tokens(node, &inputs).is_some();
        let values = program.node_value(&trace, &inputs, node).expect("the node's value");
        for (row, entries) in rows.iter().enumerate() {
            for (col, value) in entries.iter().enumerate() {
                let error = (q(values[[row, col]]) - *value).abs();
                let radius = if gathered { 0.0 } else { bands[node][[row, col]] };
                assert!(
                    to_f64(error) <= radius,
                    "node {node} row {row} col {col}: error {} beyond radius {radius}",
                    to_f64(error)
                );
                if radius > 0.0 {
                    widest_ratio = widest_ratio.max(to_f64(error) / radius);
                }
            }
        }
    }
    // The radii are bounds, not decorations: some entry uses a visible part of its band.
    assert!(widest_ratio > 1e-3, "every radius is at least 1000 times the error ({widest_ratio})");
}

#[test]
fn a_program_with_an_absent_block_executes_its_zero() {
    let mut program = fixture();
    let OperatorBody::Dense { values, precision: p, .. } = program.operators[7].body.clone() else {
        panic!("the fixture's W_out is dense")
    };
    let mut present = Array2::from_elem((5, 5), true);
    present[[2, 1]] = false;
    program.operators[7] = Arc::new(Operator::blocks(
        "W_out",
        program.operators[7].rows.clone(),
        program.operators[7].cols.clone(),
        values,
        present,
        p,
        Provenance::default(),
    )
    .expect("blocks"));
    assert_eq!(program.operators[7].matrix()[[2, 1]], 0.0);
    assert_eq!(program.operators[7].real_count(), 24);
    let trace = program.execute(&family(), false).expect("executes");
    assert!(trace.bands.is_none());
    assert_eq!(trace.values[14].dim(), (49, 5));
}

#[test]
fn prune_drops_what_the_output_does_not_read() {
    let mut program = fixture();
    program.nodes[13] = Node::Affine { terms: vec![(12, 7)], bias: None };
    let before = program.execute(&family(), false).expect("executes").values[14].clone();
    program.prune();
    assert!(program.operators.iter().all(|op| op.name != "W_direct"));
    let after = program.execute(&family(), false).expect("executes");
    assert_eq!(after.values[program.output], before);
}

#[test]
fn lattice_rounding_is_applied_at_construction() {
    let op = Operator::dense(
        "x",
        Interface::native(1).expect("interface"),
        Interface::native(2).expect("interface"),
        array![[0.3, -1.26]],
        precision(2),
        Provenance::default(),
    )
    .expect("dense");
    assert_eq!(op.matrix(), array![[0.25, -1.25]]);
}

/// A step finer than the reals' range allows (index beyond `2^53`) is coarsened to the finest the
/// range allows, at construction and when an edit moves the lattice: the program still encodes
/// and decodes (the p = 31 run at n = 10^5 refused a real of -1.56e6 at step 2^-33).
#[test]
fn a_lattice_step_is_derived_from_the_value_range() {
    let value = -1561496.4080355994;
    let op = Operator::dense(
        "x",
        Interface::native(1).expect("interface"),
        Interface::native(2).expect("interface"),
        array![[value, 0.1]],
        precision(33),
        Provenance::default(),
    )
    .expect("dense");
    let OperatorBody::Dense { precision: chosen, .. } = &op.body else { unreachable!() };
    assert_eq!(chosen.fraction_bits(), 52 - value.abs().log2().ceil() as i32);
    assert!((op.matrix()[[0, 0]] - value).abs() <= chosen.worst_case_error());
    let mut program = fixture();
    let index = program.operators.iter().position(|o| matches!(o.body, OperatorBody::Dense { .. })).expect("a dense operator");
    let largest = program.operators[index].largest_real();
    super::engine::apply_edit(&mut program, &super::engine::Edit::Precision { operator: index, precision: precision(1000) })
        .expect("the edit applies");
    let OperatorBody::Dense { precision: moved, .. } = &program.operators[index].body else { unreachable!() };
    assert_eq!(moved.fraction_bits(), 52 - largest.log2().ceil() as i32);
    let message = program.encode().expect("the program encodes");
    let decoded = OperatorProgram::decode(&message, &program.declarations).expect("the message decodes");
    assert_eq!(decoded.operators[index].matrix(), program.operators[index].matrix());
}

#[test]
fn incremental_execution_matches_a_full_execution() {
    let base = fixture();
    let inputs = family();
    let trace = base.execute(&inputs, false).expect("executes");
    let mut edited = base.clone();
    // Drop a block of W_in, move W_out to a coarser lattice, and change one unit's law.
    let w_in = edited.operators[5].clone();
    let OperatorBody::Dense { values, present, precision: q } = &w_in.body else { panic!("dense") };
    let mut present = present.clone();
    present[[2, 0]] = false;
    edited.operators[5] = Arc::new(
        Operator::blocks("W_in", w_in.rows.clone(), w_in.cols.clone(), values.clone(), present, *q, Provenance::default())
            .expect("blocks"),
    );
    let w_out = edited.operators[7].clone();
    edited.operators[7] = Arc::new(
        Operator::dense("W_out", w_out.rows.clone(), w_out.cols.clone(), w_out.matrix(), precision(3), Provenance::default())
            .expect("coarse"),
    );
    if let Node::Pointwise { laws, .. } = &mut edited.nodes[12] {
        laws[0] = Law::Zero;
    }
    let incremental = edited.execute_incremental(&inputs, &base, &trace).expect("incremental");
    let full = edited.execute(&inputs, false).expect("executes");
    let reference = &full.values[edited.output];
    let scale = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = (&incremental - reference).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(worst <= 1e-12 * scale, "incremental differs by {worst} at scale {scale}");
    assert_ne!(incremental, trace.values[base.output]);
}

#[test]
fn elias_delta_round_trips_and_its_signed_index_length_is_subadditive() {
    use super::codec::{decode_signed_delta, encode_signed_delta, signed_delta_len_bits};
    let values: Vec<i64> = (-300..300).chain([1 << 20, -(1 << 20), i64::MAX, i64::MIN + 1]).collect();
    let mut message = super::codec::BitString::new();
    let mut expected = 0;
    for &v in &values {
        encode_signed_delta(&mut message, v).expect("encodes");
        expected += signed_delta_len_bits(v).expect("length");
    }
    assert_eq!(message.len_bits(), expected);
    let mut reader = message.reader();
    for &v in &values {
        assert_eq!(decode_signed_delta(&mut reader).expect("decodes"), v);
    }
    assert_eq!(reader.remaining_bits(), 0);
    for a in -600i64..600 {
        for b in [-513i64, -64, -9, -8, -7, -1, 1, 7, 8, 9, 64, 513] {
            if a == 0 {
                continue;
            }
            let whole = signed_delta_len_bits(a + b).expect("length");
            let split = signed_delta_len_bits(a).expect("length") + signed_delta_len_bits(b).expect("length");
            assert!(whole <= split, "L({}) = {whole} > L({a}) + L({b}) = {split}", a + b);
        }
    }
}

#[test]
fn splitting_an_operator_on_its_lattice_never_shortens_the_program() {
    // One hundred entries of 8 and one of 1 on the integer lattice: under the ω index code the split
    // [7, …, 7, 1] + [1, …, 1, 0] was 99 index bits shorter; under δ it is longer.
    let declarations = Declarations { parameters: 0, domains: vec![], slots: vec![Slot::Raw { width: 101 }] };
    let input = Interface::native(101).expect("interface");
    let output = Interface::native(1).expect("interface");
    let integer = precision(0);
    let mut whole_values = Array2::<f64>::from_elem((1, 101), 8.0);
    whole_values[[0, 100]] = 1.0;
    let mut first = Array2::<f64>::from_elem((1, 101), 7.0);
    first[[0, 100]] = 1.0;
    let mut second = Array2::<f64>::from_elem((1, 101), 1.0);
    second[[0, 100]] = 0.0;
    let op = |name: &str, values: Array2<f64>| {
        Operator::dense(name, output.clone(), input.clone(), values, integer, Provenance::default()).expect("dense")
    };
    let whole = OperatorProgram { rules: Vec::new(),
        declarations: declarations.clone(),
        bases: vec![],
        operators: vec![Arc::new(op("whole", whole_values))],
        nodes: vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0)], bias: None }],
        output: 1,
    };
    let split = OperatorProgram { rules: Vec::new(),
        declarations,
        bases: vec![],
        operators: vec![Arc::new(op("first", first)), Arc::new(op("second", second))],
        nodes: vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 0), (0, 1)], bias: None }],
        output: 1,
    };
    let (whole_bits, split_bits) = (whole.code_bits().expect("bits"), split.code_bits().expect("bits"));
    assert!(split_bits > whole_bits, "split {split_bits} bits against whole {whole_bits}");
}

#[test]
fn a_rule_is_sent_once_and_executes_at_each_call_and_a_gain_reads_the_declared_parameters() {
    use super::operator_program::{Coefficient, Rule};
    let declarations =
        Declarations { parameters: 1, domains: vec![], slots: vec![Slot::Raw { width: 3 }, Slot::Raw { width: 3 }] };
    let native = Interface::native(3).expect("interface");
    let units = Interface::uniform(2, 1, LabelKind::Unit, 0).expect("interface");
    let w = Operator::dense("W", units, native.clone(), reals(2, 3, 11), precision(16), Provenance::default())
        .expect("dense");
    let rule = Rule {
        name: "unit".to_string(),
        inputs: vec![native],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Pointwise { input: 1, laws: vec![Law::Relu, Law::Relu] },
        ],
        output: 2,
    };
    let program = OperatorProgram {
        rules: vec![rule],
        declarations: declarations.clone(),
        bases: vec![],
        operators: vec![Arc::new(w.clone())],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Raw { slot: 1 },
            Node::Call { rule: 0, arguments: vec![0] },
            Node::Call { rule: 0, arguments: vec![1] },
            Node::Concat { parts: vec![2, 3] },
            Node::Gain {
                input: 4,
                coefficient: Coefficient::Product(vec![Coefficient::Parameter(0), Coefficient::Number(0.5)]),
            },
        ],
        output: 5,
    };
    let message = program.encode().expect("encodes");
    assert_eq!(message.len_bits(), program.code_bits().expect("bits"));
    let decoded = OperatorProgram::decode(&message, &declarations).expect("decodes");
    assert_eq!(decoded.nodes, program.nodes);
    assert_eq!(decoded.rules[0].nodes, program.rules[0].nodes);
    let x = Array2::from_shape_fn((4, 3), |(i, j)| (i as f64 - 1.5) * (j as f64 + 0.25));
    let y = Array2::from_shape_fn((4, 3), |(i, j)| (j as f64 - 1.0) * (i as f64 + 0.5));
    let inputs = FamilyInputs { layout: None, rows: 4, slots: vec![SlotValues::Raw(x.clone()), SlotValues::Raw(y.clone())] };
    let at_two = program.execute_at(&inputs, true, &[2.0]).expect("executes");
    let matrix = w.matrix();
    let relu = |m: Array2<f64>| m.mapv(|v| v.max(0.0));
    let expected =
        ndarray::concatenate(ndarray::Axis(1), &[relu(x.dot(&matrix.t())).view(), relu(y.dot(&matrix.t())).view()])
            .expect("concat");
    let value = &at_two.values[5];
    let bands = &at_two.band(5).expect("bands");
    for ((v, b), e) in value.iter().zip(bands.iter()).zip(expected.iter()) {
        assert!((v - e).abs() <= b + 1e-15 * e.abs(), "{v} vs {e} within {b}");
    }
    // The body's operator is paid once, however many calls read it.
    let (_, real_bits) = w.code_bits().expect("operator bits");
    let account = program.code_account().expect("account");
    assert_eq!(account.operator_bits.iter().map(|(_, r)| r).sum::<u64>(), real_bits);
}
