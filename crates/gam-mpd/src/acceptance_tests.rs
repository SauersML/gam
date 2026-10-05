#![cfg(test)]
//! The one acceptance path on small programs built to catch its shortcuts: each test names the
//! shortcut (a)–(i) it guards against.

use super::acceptance::{
    CostCache, Local, structural_cost,
};
use super::artifact::{Argument, Artifact, Callee};
use super::operator_program::{
    Declarations, FamilyInputs, Interface, Law, Node, Operator, OperatorProgram, Provenance, Rule, Slot, SlotValues, exact_precision,
};
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

fn dense(name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>) -> Operator {
    let precision = exact_precision(values.iter().copied()).expect("a lattice");
    Operator::dense(name, rows.clone(), cols.clone(), values, precision, Provenance::native(name)).expect("a dense operator")
}

fn column(name: &str, rows: &Interface, values: &[f64]) -> Operator {
    dense(name, rows, &Interface::constant(), Array2::from_shape_vec((values.len(), 1), values.to_vec()).expect("a column"))
}

fn native(width: usize) -> Interface {
    Interface::native(width).expect("an interface")
}

fn raw_program(width: usize, operators: Vec<Operator>, nodes: Vec<Node>) -> OperatorProgram {
    let output = nodes.len() - 1;
    OperatorProgram {
        declarations: Declarations { domains: Vec::new(), slots: vec![Slot::Raw { width }], parameters: 0 },
        bases: Vec::new(),
        operators: operators.into_iter().map(Arc::new).collect(),
        rules: Vec::new(),
        nodes,
        output,
    }
}

fn raw_family(rows: Vec<Vec<f64>>) -> FamilyInputs {
    let width = rows[0].len();
    let n = rows.len();
    let values = Array2::from_shape_vec((n, width), rows.into_iter().flatten().collect()).expect("rows");
    FamilyInputs { rows: n, slots: vec![SlotValues::Raw(values)], layout: None }
}

fn grid(points: &[f64], width: usize) -> FamilyInputs {
    let mut rows: Vec<Vec<f64>> = vec![Vec::new()];
    for _ in 0..width {
        rows = rows.into_iter().flat_map(|row| points.iter().map(move |p| [row.clone(), vec![*p]].concat())).collect();
    }
    raw_family(rows)
}

/// A rule of one input of interface `input` whose body is `nodes` after `Param { 0 }` (node 0).
fn rule(name: &str, inputs: Vec<Interface>, nodes: Vec<Node>) -> Rule {
    let output = nodes.len() - 1;
    Rule { name: name.to_string(), inputs, nodes, output }
}

/// A one-term rule `Param → A_k Param` writing interface `rows` from `cols`.
fn linear(start: &Artifact, name: &str, read: usize, write: usize, op: Operator) -> Artifact {
    let k = start.program.operators.len();
    let cols = op.cols.clone();
    let body = rule(name, vec![cols], vec![Node::Param { index: 0 }, Node::Affine { terms: vec![(0, k)], bias: None }]);
    start.replace_block(name, Callee::New(body), vec![Argument::Native(read)], write, vec![op]).expect("a replacement")
}

/// The counterexample ascent tests inputs beyond the declared family: a linear rule for a block
/// that adds `gelu(x₀ − 2) e` is close on the family (`x₀ ≤ 1`) and far at the box's edge, which
/// the ascent reaches from the family's rows.
#[test]
fn the_counterexample_ascent_finds_a_worse_input() {
    let (m, classes, one) = (native(2), native(3), native(1));
    let model = raw_program(
        2,
        vec![
            Operator::identity("I", m.clone()),
            dense("A", &m, &m, array![[1.0, 0.5], [-0.25, 1.0]]),
            dense("first", &one, &m, array![[1.0, 0.0]]),
            column("minus two", &one, &[-2.0]),
            dense("e", &m, &one, array![[1.0], [-0.5]]),
            dense("U", &classes, &m, array![[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]),
        ],
        vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 2)], bias: Some(3) },
            Node::Pointwise { input: 2, laws: vec![Law::GeluTanh] },
            Node::Affine { terms: vec![(1, 1), (3, 4)], bias: None },
            Node::Affine { terms: vec![(1, 0), (4, 0)], bias: None },
            Node::Affine { terms: vec![(5, 5)], bias: None },
        ],
    );
    let family = grid(&[-1.0, -0.5, 0.0, 0.5, 1.0], 2);
    let start = Artifact::native(&model).expect("an artifact");
    let candidate = linear(&start, "linear part", 1, 4, dense("A'", &m, &m, array![[1.0, 0.5], [-0.25, 1.0]]));
    let declared = Local::new(&model, family.clone(), None, 64).measure(&candidate).expect("a measure");
    let domain = vec![super::acceptance::SlotDomain::Box { lower: Array1::from_elem(2, -3.0), upper: Array1::from_elem(2, 3.0) }];
    let ascent = super::acceptance::Ascent { domain, pool: Vec::new(), evaluations: 16 };
    let searched = Local::new(&model, family.clone(), Some(ascent), 64).measure(&candidate).expect("a measure");
    assert!(searched.counterexamples > 0, "{searched:?}");
    assert!(searched.blocks[0].worst > 2.0 * declared.blocks[0].worst, "{} against {}", searched.blocks[0].worst, declared.blocks[0].worst);
    assert_eq!(searched.blocks[0].scale, declared.blocks[0].scale, "the scale is the declared family's");
}

// Append to acceptance_tests.rs.
#[test]
fn gain_literals_round_recursively_in_nodes_and_shared_rules() {
    use super::operator_program::Coefficient;
    let model = raw_program(
        2,
        vec![],
        vec![
            Node::Raw { slot: 0 },
            Node::Gain {
                input: 0,
                coefficient: Coefficient::Number(1.0),
            },
        ],
    );
    let mut candidate = Artifact::native(&model)
        .unwrap()
        .replace_block(
            "gain",
            Callee::New(rule(
                "gain",
                vec![native(2)],
                vec![
                    Node::Param { index: 0 },
                    Node::Gain {
                        input: 0,
                        coefficient: Coefficient::Product(vec![
                            Coefficient::Number(0.1),
                            Coefficient::Sum(vec![
                                Coefficient::Number(0.3),
                                Coefficient::Number(0.5),
                            ]),
                        ]),
                    },
                ],
            )),
            vec![Argument::Native(0)],
            1,
            vec![],
        )
        .unwrap();
    // Also put a non-f32 gain in the ordinary node list.
    let previous = candidate.program.output;
    candidate.program.nodes.push(Node::Gain {
        input: previous,
        coefficient: Coefficient::Number(0.7),
    });
    candidate.program.output = candidate.program.nodes.len() - 1;
    assert!(!candidate.has_f32_literals());
    let rounded = candidate.f32_literals().unwrap();
    assert!(rounded.has_f32_literals());
    let decoded = Artifact::from_bytes(&rounded.to_bytes().unwrap(), &model.declarations).unwrap();
    let family = raw_family(vec![vec![1.0, 2.0]]);
    let output = decoded.execute(&family).unwrap().values[decoded.program.output].clone();
    let expected = (0.1f32 as f64) * ((0.3f32 as f64) + 0.5) * (0.7f32 as f64);
    assert_eq!(output[[0, 0]], expected);
    assert_eq!(output[[0, 1]], expected * 2.0);
}

/// Shared numerical bodies are charged once; a call pays structure, never another copy of the literals.
#[test]
fn c32_shared_rule_real_literals_are_paid_once() {
    use super::operator_program::Coefficient;
    let body = rule("shared numerical body", vec![native(2)], vec![
        Node::Param { index: 0 },
        Node::Gain { input: 0, coefficient: Coefficient::Product(vec![
            Coefficient::Number(0.25), Coefficient::Number(0.5),
        ]) },
        Node::RmsNorm { input: 1, epsilon: 1e-5 },
    ]);
    let mut one = raw_program(2, vec![], vec![Node::Raw { slot: 0 }, Node::Call { rule: 0, arguments: vec![0] }]);
    one.rules.push(body.clone());
    let mut two = one.clone();
    two.nodes.push(Node::Call { rule: 0, arguments: vec![1] });
    two.output = 2;
    let cost = |program: &OperatorProgram| structural_cost(&Artifact::native(program).unwrap(), &mut CostCache::default()).unwrap();
    let (a, b) = (cost(&one), cost(&two));
    assert_eq!((a.literals, b.literals), (3, 3));
    assert!(b.structure_bits > a.structure_bits, "the extra call and its wiring still cost structure");
    let mut duplicated = two;
    duplicated.rules.push(body);
    duplicated.nodes[2] = Node::Call { rule: 1, arguments: vec![1] };
    assert_eq!(cost(&duplicated).literals, 6, "separately stored bodies have independent payloads");
}

/// A pricing correction must neither round a native architecture epsilon nor change its decoded execution.
#[test]
fn c32_preserves_exact_native_epsilon_and_execution() {
    let epsilon = 1e-5;
    assert_ne!(epsilon, f64::from(epsilon as f32));
    let program = raw_program(2, vec![], vec![Node::Raw { slot: 0 }, Node::RmsNorm { input: 0, epsilon }]);
    let artifact = Artifact::native(&program).unwrap().f32_literals().unwrap();
    let decoded = Artifact::from_bytes(&artifact.to_bytes().unwrap(), &program.declarations).unwrap();
    assert_eq!(decoded.program.nodes, program.nodes, "wire decoding preserves the exact architecture value");
    let family = raw_family(vec![vec![0.25, 2.0], vec![-0.5, 0.75]]);
    assert_eq!(decoded.execute(&family).unwrap().values, program.execute(&family, false).unwrap().values);
    decoded.validate_coverage(&program).unwrap();
    assert_eq!(structural_cost(&decoded, &mut CostCache::default()).unwrap().literals, 1);
}

/// Indicator values are fixed primitives; the domain's finite size is structural.
#[test]
fn c32_indicator_basis_has_no_independent_numerical_literals() {
    use super::operator_program::{Basis, Domain};
    let program = |size| OperatorProgram {
        declarations: Declarations { domains: vec![Domain { size }], slots: vec![Slot::Token { domain: 0 }], parameters: 0 },
        bases: vec![Basis::Indicator { domain: 0 }],
        operators: vec![], rules: vec![],
        nodes: vec![Node::Feature { slot: 0, basis: 0 }, Node::Readout { input: 0, basis: 0 }], output: 1,
    };
    for size in [3, 17] {
        let p = program(size);
        let cost = structural_cost(&Artifact::native(&p).unwrap(), &mut CostCache::default()).unwrap();
        assert_eq!(cost.literals, 0);
        assert!(cost.structure_bits > 0);
    }
}

/// A bilinear scale equal to input width still lacks an explicit dimension derivation in the current enum.
#[test]
fn c32_bilinear_scale_argument_is_not_free_when_equal_to_width() {
    use super::operator_program::Scale;
    let make = |n| raw_program(2, vec![], vec![Node::Raw { slot: 0 }, Node::Bilinear { left: 0, right: 0, scale: Scale::InverseSqrt(n) }]);
    let cost = |p: &OperatorProgram| structural_cost(&Artifact::native(p).unwrap(), &mut CostCache::default()).unwrap();
    assert_eq!(cost(&make(2)).literals, 1);
    assert_eq!(cost(&make(2)), cost(&make(63)));
}

/// Dimensions specify structure even when the same node also has priced arithmetic literals.
#[test]
fn c32_rotary_dimension_is_structure_not_an_extra_numeric_literal() {
    use super::operator_program::{Rotary, Scale};
    let make = |dims| raw_program(4, vec![], vec![Node::Raw { slot: 0 }, Node::Attend {
        query: 0, key: 0, value: 0, scale: Scale::One,
        rotary: Some(Rotary { base: 10000, dims, half_split: true }), causal: true,
    }]);
    let cost = |p: &OperatorProgram| structural_cost(&Artifact::native(p).unwrap(), &mut CostCache::default()).unwrap();
    let (a, b) = (cost(&make(2)), cost(&make(4)));
    assert_eq!((a.literals, b.literals), (1, 1));
    assert_ne!(a.structure_bits, b.structure_bits, "the chosen dimension remains explicitly coded structure");
}

#[path = "local_first_tests.rs"]
mod local_first_tests;

