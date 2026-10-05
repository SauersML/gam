#![cfg(test)]
//! Operator programs: the message decodes to the program and has the computed length, and every
//! banded execution's radius covers the quad-double evaluation of the same program.

use super::operator_program::{
    Basis, Declarations, Domain, FamilyInputs, Interface, LabelKind, Law, Node, Operator, OperatorBody, OperatorProgram,
    Provenance, Scale, Slot, SlotValues,
};
use super::precision::DeclaredPrecision;
use ndarray::{Array2, array};
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
            Domain { size: 7 },
            Domain { size: 5 },
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

/// A mostly-zero argument's product over its nonzeros is the dense product's, within the float64
/// summation bound of the same products in another order (`2 γ_{k}` of `|x| |A|ᵀ`).
#[test]
fn a_sparse_arguments_product_is_the_dense_one() {
    use super::operator_program::{sparse_abt, sparse_enough};
    let noise = |seed: usize| {
        let mut v = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        v ^= v >> 33;
        v = v.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        v ^= v >> 33;
        (v >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    };
    let (rows, inner, outer) = (37, 400, 23);
    let x = Array2::from_shape_fn((rows, inner), |(r, c)| if noise(r * inner + c) > 0.9 { noise(7 + r * inner + c) } else { 0.0 });
    let a = Array2::from_shape_fn((outer, inner), |(o, c)| noise(100_000 + o * inner + c));
    assert!(sparse_enough(&x));
    let (sparse, dense) = (sparse_abt(&x, &a), x.dot(&a.t()));
    let magnitude = x.mapv(f64::abs).dot(&a.mapv(f64::abs).t());
    let k = inner as f64 * f64::EPSILON / 2.0;
    for ((s, d), m) in sparse.iter().zip(dense.iter()).zip(magnitude.iter()) {
        assert!((s - d).abs() <= 2.0 * k / (1.0 - k) * m, "{s} against {d}");
    }
    assert!(!sparse_enough(&Array2::ones((3, 3))));
}

#[test]
fn indicator_reads_check_domains_widths_and_radius_shapes() {
    let declarations = Declarations { domains: vec![Domain { size: 2 }], slots: vec![], parameters: 0 };
    let basis = Basis::Indicator { domain: 0 };
    let values = Array2::ones((3, 2));
    assert_eq!(basis.read(&declarations, &values).unwrap(), values);
    assert_eq!(basis.read_transpose(&declarations, &values).unwrap(), values);
    assert!(basis.read(&declarations, &Array2::ones((3, 1))).is_err());
    assert!(Basis::Indicator { domain: 1 }.read(&declarations, &values).is_err());
    assert!(basis.read_banded(&declarations, &values, Some(&Array2::zeros((1, 2)))).is_err());
}

#[test]
fn operator_structure_price_matches_wire_without_coding_numerical_payloads() {
    let program = fixture();
    let mut operators: Vec<Operator> = program.operators.iter().map(|o| (**o).clone()).collect();
    let interface = Interface::uniform(3, 1, LabelKind::Native, 0).unwrap();
    operators.push(Operator::identity("identity", interface.clone()));
    operators.push(Operator::diag("diagonal", interface.clone(), ndarray::array![1.0, -0.25, 0.0], precision(8), Provenance::default()).unwrap());
    operators.push(Operator::low_rank("rank", interface.clone(), interface, Array2::ones((3, 1)), Array2::ones((1, 3)), precision(8), Provenance::default()).unwrap());
    for operator in operators {
        assert_eq!(operator.structure_bits().unwrap(), operator.code_bits().unwrap().0, "{}", operator.name);
    }
}

