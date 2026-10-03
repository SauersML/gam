//! The printer on a planted two-frequency circuit: one rule, its bindings, fingerprints up to
//! change of basis, the bit split and the behaviour map.

use std::sync::Arc;
use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorProgram, Provenance,
    Rule, Slot, SlotValues,
};
use super::precision::DeclaredPrecision;
use super::printer::{Behaviour, print};
use ndarray::Array2;

/// Units 0, 1 read plane 1 of both tokens and write plane 1; units 2, 3 are the same circuit on
/// plane 2, every block turned by a quarter rotation and each unit rescaled by 2 (in) and 1/2
/// (out). `perturb` moves one real of unit 3 off that image.
fn planted(perturb: bool) -> OperatorProgram {
    let characters = |domain| Basis::Characters { domain, positions: (0..5).map(Some).collect(), declared: true };
    let declarations = Declarations {
        domains: vec![Domain { size: 5, cycle: Some((0..5).map(Some).collect()) }; 2],
        slots: vec![Slot::Token { domain: 0 }, Slot::Token { domain: 0 }],
        parameters: 0,
    };
    let planes = characters(0).interface(&declarations).expect("a valid planted fixture");
    let units = Interface::uniform(4, 1, LabelKind::Unit, 0).expect("a valid planted fixture");
    let precision = DeclaredPrecision::new(4).expect("a valid planted fixture");
    // Plane `k` occupies columns `2k - 1, 2k`; a quarter turn maps (x, y) to (-y, x).
    let turn = |v: [f64; 2], scale: f64| [-v[1] * scale, v[0] * scale];
    let reads = |first: [[f64; 2]; 2], last_y: f64| {
        let mut m = Array2::<f64>::zeros((4, 5));
        for (u, row) in first.iter().enumerate() {
            m[[u, 1]] = row[0];
            m[[u, 2]] = row[1];
            let image = turn(*row, 2.0);
            m[[u + 2, 3]] = image[0];
            m[[u + 2, 4]] = image[1];
        }
        if perturb {
            m[[3, 4]] = last_y;
        }
        m
    };
    let mut read_present = Array2::from_elem((4, 3), false);
    for u in 0..4 {
        read_present[[u, if u < 2 { 1 } else { 2 }]] = true;
    }
    let native = Provenance::native;
    let wa = Operator::blocks("Wa", units.clone(), planes.clone(), reads([[1.0, 0.5], [0.25, -1.0]], 0.0), read_present.clone(), precision, native("Wa")).expect("a valid planted fixture");
    let wb = Operator::blocks("Wb", units.clone(), planes.clone(), reads([[0.75, 0.25], [-0.5, 0.5]], -0.75), read_present, precision, native("Wb")).expect("a valid planted fixture");
    let bias = Operator::dense(
        "b",
        units.clone(),
        Interface::constant(),
        Array2::from_shape_vec((4, 1), vec![0.5, -0.25, 1.0, -0.5]).expect("a valid planted fixture"),
        precision,
        native("b"),
    )
    .expect("a valid planted fixture");
    let mut out = Array2::<f64>::zeros((5, 4));
    for (u, col) in [[1.0, -0.5], [0.5, 0.25]].iter().enumerate() {
        out[[1, u]] = col[0];
        out[[2, u]] = col[1];
        let image = turn(*col, 0.5);
        out[[3, u + 2]] = image[0];
        out[[4, u + 2]] = image[1];
    }
    let mut out_present = Array2::from_elem((3, 4), false);
    for u in 0..4 {
        out_present[[if u < 2 { 1 } else { 2 }, u]] = true;
    }
    let w_out = Operator::blocks("Wout", planes.clone(), units, out, out_present, precision, native("Wout")).expect("a valid planted fixture");
    OperatorProgram {
        declarations,
        bases: vec![characters(0), characters(1)],
        operators: [wa, wb, bias, w_out].into_iter().map(Arc::new).collect(),
        rules: Vec::new(),
        nodes: vec![
            Node::Feature { slot: 0, basis: 0 },
            Node::Feature { slot: 1, basis: 0 },
            Node::Affine { terms: vec![(0, 0), (1, 1)], bias: Some(2) },
            Node::Pointwise { input: 2, laws: vec![Law::Relu; 4] },
            Node::Affine { terms: vec![(3, 3)], bias: None },
            Node::Readout { input: 4, basis: 1 },
        ],
        output: 5,
    }
}

#[test]
fn two_frequencies_are_one_rule_up_to_change_of_basis() {
    let program = planted(false);
    let printout = print(&program, None).expect("the program prints");
    assert_eq!(printout.bits.program(), program.code_bits().expect("the program has a code"));
    assert_eq!(printout.rules.len(), 1, "{printout}");
    let rule = &printout.rules[0];
    assert_eq!(rule.instances.len(), 2);
    assert_eq!(rule.distinct.len(), 1, "{printout}");
    assert!(rule.spread < 1e-12, "{}", rule.spread);
    let account = program.code_account().expect("the program has a code");
    // Every block is in the rule; each operator's lattice header (count, precision) is not.
    assert_eq!(printout.bits.table_storage, 0);
    assert!(rule.bits > 0 && rule.bits < printout.bits.other_operator_reals);
    assert_eq!(rule.reals, program.real_count());
    assert_eq!(rule.template.last().expect("a template has a root line"), "plane ← Σ_unit W·relu(W·plane(x0) + W·plane(x1) + b)");
    let labels: Vec<&str> = rule.instances.iter().map(|i| i.labels.as_str()).collect();
    assert_eq!(labels, ["Plane{1} Unit{0,1}", "Plane{2} Unit{2,3}"]);
    assert_eq!(printout.unchanged_native_bits, account.operator_bits.iter().map(|(s, r)| s + r).sum::<u64>());
    assert!(printout.bases[0].contains("declared 5-cycle, a(t) = t"), "{}", printout.bases[0]);
}

#[test]
fn a_moved_real_splits_the_rule() {
    let printout = print(&planted(true), None).expect("the program prints");
    let rule = &printout.rules[0];
    assert_eq!(rule.instances.len(), 2);
    assert_eq!(rule.distinct.len(), 2, "{printout}");
    assert!(rule.spread > 0.0);
}

#[test]
fn the_behaviour_map_accounts_every_kl_bit() {
    let program = planted(false);
    let a: Vec<u32> = (0..25).map(|r| r / 5).collect();
    let b: Vec<u32> = (0..25).map(|r| r % 5).collect();
    let inputs = FamilyInputs { rows: 25, slots: vec![SlotValues::Tokens(a.clone()), SlotValues::Tokens(b)], layout: None };
    let row_bits: Vec<f64> = a.iter().map(|&t| f64::from(t)).collect();
    let mut agrees = vec![true; 25];
    agrees[24] = false;
    let printout = print(&program, Some(&Behaviour { inputs: &inputs, row_bits: &row_bits, argmax_agrees: &agrees })).expect("the program prints");
    let map = printout.behaviour.as_ref().expect("a behaviour was given");
    assert_eq!(map.kl_bits, 50.0);
    assert_eq!(printout.bits.kl, Some(50.0));
    assert_eq!(map.by_slot[0].as_ref().expect("a token slot"), &vec![0.0, 5.0, 10.0, 15.0, 20.0]);
    assert_eq!(map.by_slot[1].as_ref().expect("a token slot").iter().sum::<f64>(), 50.0);
    // The five rows of a = 4 carry 20 bits and the five of a = 3 another 15: half needs 7 rows.
    assert_eq!(map.half, 7);
    assert_eq!(map.disagreements, 1);
    assert_eq!(map.order[0], 20);
    let text = printout.to_string();
    assert!(text.contains("worst: (4, 0) 4.00"), "{text}");
}

#[test]
fn a_called_rule_is_read_per_call_with_its_body_shared() {
    // The body `relu(W x)` over plane 1, called on each token's characters; `V` sums the calls.
    let mut program = planted(false);
    let planes = program.bases[0].interface(&program.declarations).expect("a valid planted fixture");
    let units = Interface::uniform(2, 1, LabelKind::Unit, 0).expect("a valid planted fixture");
    let precision = DeclaredPrecision::new(4).expect("a valid planted fixture");
    let mut read = Array2::from_elem((2, 3), false);
    read.column_mut(1).fill(true);
    let w = Array2::from_shape_vec((2, 5), vec![0.0, 1.0, 0.5, 0.0, 0.0, 0.0, -0.25, 1.0, 0.0, 0.0]).expect("a valid planted fixture");
    let mut write = Array2::from_elem((3, 2), false);
    write.row_mut(1).fill(true);
    let v = Array2::from_shape_vec((5, 2), vec![0.0, 0.0, 1.0, 0.5, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0]).expect("a valid planted fixture");
    program.operators = [
        Operator::blocks("W", units.clone(), planes.clone(), w, read, precision, Provenance::native("W")).expect("a valid planted fixture"),
        Operator::blocks("V", planes.clone(), units, v, write, precision, Provenance::native("V")).expect("a valid planted fixture"),
    ]
    .into_iter()
    .map(Arc::new)
    .collect();
    program.rules = vec![Rule {
        name: "pair".to_string(),
        inputs: vec![planes],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Pointwise { input: 1, laws: vec![Law::Relu; 2] },
        ],
        output: 2,
    }];
    program.nodes = vec![
        Node::Feature { slot: 0, basis: 0 },
        Node::Feature { slot: 1, basis: 0 },
        Node::Call { rule: 0, arguments: vec![0] },
        Node::Call { rule: 0, arguments: vec![1] },
        Node::Affine { terms: vec![(2, 1), (3, 1)], bias: None },
        Node::Readout { input: 4, basis: 1 },
    ];
    program.output = 5;
    let printout = print(&program, None).expect("the program prints");
    assert_eq!(printout.calls, ["pair ×2, body of 3 nodes"]);
    let unit = printout.rules.iter().find(|r| r.instances.len() == 4).unwrap_or_else(|| panic!("{printout}"));
    assert_eq!(unit.template.last().expect("a template has a root line"), "unit ← relu(W·plane(x0))");
    assert_eq!(unit.shared, ["W"]);
    // One unit read alone is one rule up to a change of basis of its plane and its scale.
    assert_eq!(unit.distinct.len(), 1, "{printout}");
    let calls: Vec<bool> = unit.instances.iter().map(|i| i.root.contains(" in pair#")).collect();
    assert_eq!(calls, [true; 4]);
}
