use super::*;
use ndarray::array;

fn relation() -> MatrixRule {
    MatrixRule {
        inputs: vec![
            Type::Matrix { rows: 2, cols: 2 },
            Type::Vector { len: 2 },
            Type::Vector { len: 2 },
        ],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Param { index: 1 },
            Node::Param { index: 2 },
            Node::Divide {
                numerator: 1,
                denominator: 2,
            },
            Node::Diag { input: 3 },
            Node::Pinv {
                input: 0,
                convention: PinvConvention::SvdResolutionBand,
            },
            Node::MatMul { left: 4, right: 5 },
        ],
        output: 6,
    }
}
#[test]
fn transpose_reciprocal_scale_and_signed_zero_are_serialized() {
    let rule = MatrixRule {
        inputs: vec![Type::Matrix { rows: 1, cols: 2 }],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Reciprocal { input: 0 },
            Node::Transpose { input: 1 },
            Node::Scale {
                input: 2,
                coefficient: -0.0,
            },
        ],
        output: 3,
    };
    let bytes = rule.encode().unwrap();
    let decoded = MatrixRule::decode(&bytes).unwrap();
    let Node::Scale { coefficient, .. } = decoded.nodes[3] else {
        panic!("scale")
    };
    assert_eq!(coefficient.to_bits(), (-0.0_f32).to_bits());
    assert_eq!(decoded.encode().unwrap(), bytes);
    assert_eq!(decoded.cost().unwrap().literals, 1);
    assert_eq!(
        decoded.cost().unwrap().structure_bits + 32,
        bytes.len_bits()
    );
    let Value::Matrix(value) = decoded
        .evaluate(&[Value::Matrix(array![[2.0, 4.0]])])
        .unwrap()
    else {
        panic!("matrix")
    };
    assert_eq!(value.dim(), (2, 1));
    assert!(value.iter().all(|v| v.to_bits() == (-0.0_f64).to_bits()));
    let mut positive = rule.clone();
    positive.nodes[3] = Node::Scale {
        input: 2,
        coefficient: 0.0,
    };
    assert_ne!(
        positive.encode().unwrap(),
        bytes,
        "pool deduplication must compare encoded bodies, not float PartialEq"
    );
}
#[test]
fn typed_graph_and_nonfinite_failures_are_explicit() {
    let mut rule = relation();
    rule.nodes[6] = Node::MatMul { left: 6, right: 5 };
    assert!(rule.types().is_err());
    assert!(rule.encode().is_err());
    rule = relation();
    rule.inputs[0] = Type::Matrix { rows: 2, cols: 3 };
    assert!(rule.types().is_err());
    rule = relation();
    rule.nodes[4] = Node::Diag { input: 0 };
    assert!(rule.types().is_err());
    let reciprocal = MatrixRule {
        inputs: vec![Type::Vector { len: 1 }],
        nodes: vec![Node::Param { index: 0 }, Node::Reciprocal { input: 0 }],
        output: 1,
    };
    for value in [0.0, -0.0, f64::NAN, f64::INFINITY] {
        assert!(
            reciprocal
                .evaluate(&[Value::Vector(array![value])])
                .is_err()
        );
    }
    assert!(
        reciprocal
            .evaluate(&[Value::Vector(array![1.0, 2.0])])
            .is_err()
    );
    let mut invalid = reciprocal.clone();
    invalid.nodes[1] = Node::Scale {
        input: 0,
        coefficient: f32::NAN,
    };
    assert!(invalid.encode().is_err());
    let zero_pinv = MatrixRule {
        inputs: vec![Type::Matrix { rows: 2, cols: 3 }],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Pinv {
                input: 0,
                convention: PinvConvention::SvdResolutionBand,
            },
        ],
        output: 1,
    };
    assert_eq!(
        zero_pinv
            .evaluate(&[Value::Matrix(Array2::zeros((2, 3)))])
            .unwrap(),
        Value::Matrix(Array2::zeros((3, 2)))
    );
}
#[test]
fn malformed_truncated_and_nonfinite_messages_are_rejected() {
    let bytes = relation().encode().unwrap();
    for length in 0..bytes.len_bits() {
        assert!(MatrixRule::decode(&bytes.reader().read_bit_string(length).unwrap()).is_err());
    }
    let mut extra = bytes.clone();
    extra.push_bit(false);
    assert!(MatrixRule::decode(&extra).is_err());
    // A syntactically coded Diag of a matrix is rejected by decoder type checking.
    let mut message = BitString::new();
    encode_prefix_integer(&mut message, VERSION).unwrap();
    encode_prefix_integer(&mut message, 2).unwrap();
    encode_fixed_index(&mut message, 0, 2).unwrap();
    encode_prefix_integer(&mut message, 1).unwrap();
    encode_prefix_integer(&mut message, 1).unwrap();
    encode_prefix_integer(&mut message, 2).unwrap();
    encode_fixed_index(&mut message, 0, NODE_KINDS).unwrap();
    encode_fixed_index(&mut message, 3, NODE_KINDS).unwrap();
    encode_fixed_index(&mut message, 1, 2).unwrap();
    assert!(MatrixRule::decode(&message).unwrap_err().contains("vector"));
    let valid = MatrixRule {
        inputs: vec![Type::Vector { len: 1 }],
        nodes: vec![
            Node::Param { index: 0 },
            Node::Scale {
                input: 0,
                coefficient: 1.0,
            },
        ],
        output: 1,
    };
    let bytes = valid.encode().unwrap();
    let tail = 33; // coefficient plus one output-index bit
    let mut corrupt = bytes
        .reader()
        .read_bit_string(bytes.len_bits() - tail)
        .unwrap();
    corrupt
        .push_bits(u64::from(f32::INFINITY.to_bits()), 32)
        .unwrap();
    corrupt.push_bit(true);
    assert!(MatrixRule::decode(&corrupt).is_err());
}
