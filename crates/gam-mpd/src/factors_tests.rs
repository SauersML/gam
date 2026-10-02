#![cfg(test)]
//! The shared-factor fit on a planted layer: units that each read one of three planes of a wider
//! residual are found as three rules, without the planes being named.

use super::contract::{Contract, FamilyKind};
use super::engine::{Budget, Coarsen, Primitive, decompose};
use super::factors::Factors;
use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram,
    Provenance, Slot, SlotValues,
};
use super::precision::DeclaredPrecision;
use ndarray::Array2;
use std::collections::BTreeSet;
use std::f64::consts::TAU;
use std::sync::Arc;

const P: usize = 11;
const WIDTH: usize = 16;
const UNITS: usize = 24;
const PLANES: [usize; 3] = [1, 2, 4];

fn precision() -> DeclaredPrecision {
    DeclaredPrecision::new(30).expect("a precision")
}

/// A deterministic value in `[-1, 1)`.
fn noise(seed: usize) -> f64 {
    let mut x = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// Two tokens over `Z_11`, embedded into a 16-wide residual through three character planes mixed
/// by a fixed map; a ReLU layer of 24 units, unit `n` reading plane `n mod 3` only (through the
/// mix) and writing that plane's class characters; logits over 11 classes.
fn planted() -> (OperatorProgram, Contract) {
    let declarations = Declarations {
        parameters: 0,
        domains: vec![Domain { size: P, cycle: None }, Domain { size: P, cycle: None }],
        slots: vec![Slot::Token { domain: 0 }, Slot::Token { domain: 0 }],
    };
    let tokens = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let residual = Interface::native(WIDTH).expect("interface");
    let units = Interface::uniform(UNITS, 1, LabelKind::Unit, 0).expect("interface");
    let classes = Interface::uniform(P, 1, LabelKind::Token, 0).expect("interface");
    let angle = |k: usize, t: usize| TAU * ((k * t) % P) as f64 / P as f64;
    // The six plane coordinates of a token, mixed into the residual.
    let mix = Array2::from_shape_fn((WIDTH, 6), |(i, j)| noise(100 + 6 * i + j));
    let characters = Array2::from_shape_fn((6, P), |(j, t)| {
        let k = PLANES[j / 2];
        if j % 2 == 0 { angle(k, t).cos() } else { angle(k, t).sin() }
    });
    let embedding = mix.dot(&characters);
    // Unit n reads plane n mod 3: its row is a phase in that plane, pulled back through the mix.
    let pseudo = {
        let gram = mix.t().dot(&mix);
        let inverse = super::dense::solve(gram.view(), Array2::eye(6).view()).expect("solve");
        inverse.dot(&mix.t())
    };
    let mut reads = Array2::<f64>::zeros((UNITS, WIDTH));
    let mut writes = Array2::<f64>::zeros((P, UNITS));
    for n in 0..UNITS {
        let plane = n % 3;
        let phase = TAU * noise(n);
        let amplitude = 1.0 + 0.5 * noise(50 + n).abs();
        let mut row = ndarray::Array1::<f64>::zeros(6);
        row[2 * plane] = amplitude * phase.cos();
        row[2 * plane + 1] = amplitude * phase.sin();
        reads.row_mut(n).assign(&row.dot(&pseudo));
        let k = PLANES[plane];
        for c in 0..P {
            writes[[c, n]] = 0.7 * (angle(k, c) - phase).cos();
        }
    }
    let bias = Array2::from_shape_fn((UNITS, 1), |(n, _)| -0.3 + 0.1 * noise(200 + n));
    let operators = vec![
        Operator::dense("E", residual.clone(), tokens, embedding, precision(), Provenance::native("E")).expect("dense"),
        Operator::dense("W_in", units.clone(), residual, reads, precision(), Provenance::native("W_in")).expect("dense"),
        Operator::dense("b_in", units.clone(), Interface::constant(), bias, precision(), Provenance::native("b_in")).expect("dense"),
        Operator::dense("W_out", classes, units, writes, precision(), Provenance::native("W_out")).expect("dense"),
    ];
    let nodes = vec![
        Node::Feature { slot: 0, basis: 0 },
        Node::Feature { slot: 1, basis: 0 },
        Node::Affine { terms: vec![(0, 0), (1, 0)], bias: None },
        Node::Affine { terms: vec![(2, 1)], bias: Some(2) },
        Node::Pointwise { input: 3, laws: vec![Law::Relu; UNITS] },
        Node::Affine { terms: vec![(4, 3)], bias: None },
        Node::Readout { input: 5, basis: 1 },
    ];
    let program = OperatorProgram {
        declarations: declarations.clone(),
        bases: vec![Basis::Indicator { domain: 0 }, Basis::Indicator { domain: 1 }],
        operators: operators.into_iter().map(Arc::new).collect(),
        rules: Vec::new(),
        nodes,
        output: 6,
    };
    let pairs: Vec<(u32, u32)> = (0..P as u32).flat_map(|a| (0..P as u32).map(move |b| (a, b))).collect();
    let contract = Contract {
        declarations,
        family: FamilyInputs {
            layout: None,
            rows: pairs.len(),
            slots: vec![
                SlotValues::Tokens(pairs.iter().map(|q| q.0).collect()),
                SlotValues::Tokens(pairs.iter().map(|q| q.1).collect()),
            ],
        },
        kind: FamilyKind::Complete { description: "Z_11^2".to_string() },
        observations: 10_000,
        readouts: 1,
        readout_slots: None,
    };
    (program, contract)
}

#[test]
fn units_reading_one_plane_each_come_out_as_one_rule_per_plane() {
    let (program, contract) = planted();
    let library: Vec<Box<dyn Primitive>> = vec![Box::new(Factors), Box::new(Coarsen)];
    let budget = Budget { screenings: 10_000, certifications: 200, refit: None };
    let result = decompose(&program, &contract, &library, &budget).expect("decomposes");
    let native = contract.score(&program, &contract.logits(&program).expect("reference")).expect("native");
    assert!(result.score.proven_shorter_than(&native), "{} against native {}", result.score.total(), native.total());
    // The units' reads: the coefficient operator from the factor coordinates into the units.
    let coefficients = result
        .program
        .operators
        .iter()
        .find(|op| {
            op.cols.groups().iter().all(|g| g.label.kind == LabelKind::Factor)
                && op.rows.groups().iter().all(|g| g.label.kind == LabelKind::Unit)
        })
        .expect("the units read shared factors");
    let OperatorBody::Dense { present, .. } = &coefficients.body else { panic!("dense coefficients") };
    let mut supports: BTreeSet<Vec<usize>> = BTreeSet::new();
    for (n, row) in present.outer_iter().enumerate() {
        let support: Vec<usize> = row.iter().enumerate().filter(|(_, k)| **k).map(|(i, _)| i).collect();
        assert!(
            support.len() <= 2,
            "unit {n} reads {support:?} of {}: {:?}; curve {:?}",
            coefficients.cols.width(),
            coefficients.matrix().row(n),
            result.curve
        );
        supports.insert(support);
    }
    // Three rules: the factors the units read together fall into three disjoint groups (a unit
    // whose phase lies on an axis reads one coordinate of its plane).
    let mut group: Vec<usize> = (0..coefficients.cols.width()).collect();
    fn root(group: &mut [usize], mut x: usize) -> usize {
        while group[x] != x {
            x = group[x];
        }
        x
    }
    for support in &supports {
        for pair in support.windows(2) {
            let (a, b) = (root(&mut group, pair[0]), root(&mut group, pair[1]));
            group[a] = b;
        }
    }
    let read: BTreeSet<usize> = supports.iter().flatten().copied().collect();
    let rules: BTreeSet<usize> = read.iter().map(|&f| root(&mut group, f)).collect();
    assert_eq!(rules.len(), 3, "{supports:?}");
    // Its gauge class names the factor coordinates' GL, the units' permutations and scales, and the
    // softmax shift; every tie is classified.
    let identification = super::identify::identify(&program, &contract, &result).expect("identifies");
    assert!(identification.gauge.iter().any(|g| g.contains("GL(")), "{:?}", identification.gauge);
    assert!(identification.gauge.iter().any(|g| g.contains("softmax shift")), "{:?}", identification.gauge);
    assert_eq!(identification.alternatives.len(), result.ties.len());
}

#[test]
fn the_reverse_pass_is_the_transpose_of_the_forward_pass() {
    let (program, contract) = planted();
    let trace = program.execute(&contract.family, false).expect("executes");
    let rows = contract.family.rows;
    let output = &trace.values[program.output];
    let cotangent = Array2::from_shape_fn(output.dim(), |(r, c)| noise(1000 + 37 * r + c));
    let back = super::derivatives::vjp(&program, &contract.family, &trace, cotangent.clone()).expect("reverse");
    // A tangent of W_in moves the pre-activation node (3) by x dAᵀ, with x node 2.
    let tangent = Array2::from_shape_fn((UNITS, WIDTH), |(r, c)| noise(5000 + 17 * r + c));
    let tangents = [(1usize, tangent.clone())].into_iter().collect();
    let forward = super::derivatives::jvp(&program, &contract.family, &trace, &tangents).expect("forward");
    let left: f64 = cotangent.iter().zip(forward.iter()).map(|(a, b)| a * b).sum();
    let moved = trace.values[2].dot(&tangent.t());
    let right: f64 = back[3].as_ref().expect("a cotangent at the pre-activation").iter().zip(moved.iter()).map(|(a, b)| a * b).sum();
    assert!((left - right).abs() <= 1e-9 * left.abs().max(1.0), "{left} against {right} over {rows} rows");
}

#[test]
fn the_reverse_pass_through_rotary_causal_attention_is_the_transpose_of_the_forward_pass() {
    use super::operator_program::{Rotary, Scale, SequenceLayout};
    let width = 4;
    let rows = 8;
    let x = Array2::from_shape_fn((rows, width), |(r, c)| noise(300 + 5 * r + c));
    let native = Interface::native(width).expect("interface");
    let operators: Vec<Operator> = (0..3)
        .map(|k| {
            let m = Array2::from_shape_fn((width, width), |(r, c)| noise(400 + 31 * k + 7 * r + c));
            Operator::dense(format!("W{k}"), native.clone(), native.clone(), m, precision(), Provenance::default()).expect("dense")
        })
        .collect();
    let program = OperatorProgram {
        declarations: Declarations { parameters: 0, domains: Vec::new(), slots: vec![Slot::Raw { width }] },
        bases: Vec::new(),
        operators: operators.into_iter().map(Arc::new).collect(),
        rules: Vec::new(),
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(0, 1)], bias: None },
            Node::Affine { terms: vec![(0, 2)], bias: None },
            Node::Attend {
                query: 1,
                key: 2,
                value: 3,
                scale: Scale::InverseSqrt(width as u32),
                rotary: Some(Rotary { base: 10_000, dims: width as u32, half_split: false }),
                causal: true,
            },
        ],
        output: 4,
    };
    let family = FamilyInputs {
        layout: Some(SequenceLayout { sequence: (0..rows as u32).map(|r| r / 4).collect(), position: (0..rows as u32).map(|r| r % 4).collect() }),
        rows,
        slots: vec![SlotValues::Raw(x.clone())],
    };
    let trace = program.execute(&family, false).expect("executes");
    let cotangent = Array2::from_shape_fn((rows, width), |(r, c)| noise(700 + 11 * r + c));
    let back = super::derivatives::vjp(&program, &family, &trace, cotangent.clone()).expect("reverse");
    let tangents: std::collections::BTreeMap<usize, Array2<f64>> =
        (0..3).map(|k| (k, Array2::from_shape_fn((width, width), |(r, c)| noise(900 + 13 * k + 3 * r + c)))).collect();
    let forward = super::derivatives::jvp(&program, &family, &trace, &tangents).expect("forward");
    let left: f64 = cotangent.iter().zip(forward.iter()).map(|(a, b)| a * b).sum();
    let right: f64 = (0..3)
        .map(|k| {
            let moved = x.dot(&tangents[&k].t());
            back[k + 1].as_ref().expect("a cotangent").iter().zip(moved.iter()).map(|(a, b)| a * b).sum::<f64>()
        })
        .sum();
    assert!((left - right).abs() <= 1e-9 * left.abs().max(1.0), "{left} against {right}");
}
