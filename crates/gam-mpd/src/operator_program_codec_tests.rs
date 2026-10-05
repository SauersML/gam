//! The parallel operator codec writes and reads exactly the sequential message: fast Elias δ
//! codewords match the bitwise ones, chunked lattices match one pass, chunked and skipped reads
//! match the sequential read, and moved native codewords decode to the ordinary operators.
use super::*;

/// The δ codeword written bit group by bit group, as the format defines it.
fn reference_delta(out: &mut BitString, value: u64) {
    let low = u64::BITS - 1 - value.leading_zeros();
    let length = u64::from(low) + 1;
    let length_bits = u64::BITS - length.leading_zeros();
    for _ in 1..length_bits {
        out.push_bit(false);
    }
    out.push_bits(length, length_bits).expect("length field");
    out.push_bits(value & ((1u64 << low) - 1), low).expect("low bits");
}

fn delta_values() -> Vec<u64> {
    let mut values: Vec<u64> = (1..2000).collect();
    for shift in 1..64 {
        let top = 1u64 << shift;
        values.extend([top - 1, top, top + 1, top | (top >> 1), top | 0x5555_5555_5555_5555 & (top - 1)]);
    }
    values.push(u64::MAX);
    values
}

#[test]
fn fast_elias_delta_codewords_are_the_bitwise_ones() {
    use crate::codec::{decode_elias_delta, encode_elias_delta, skip_elias_delta};
    let values = delta_values();
    // Unaligned starts exercise every bit offset of the word peeks.
    for lead in 0..9 {
        let mut fast = BitString::new();
        let mut reference = BitString::new();
        for _ in 0..lead {
            fast.push_bit(true);
            reference.push_bit(true);
        }
        for &v in &values {
            encode_elias_delta(&mut fast, v).expect("fast codeword");
            reference_delta(&mut reference, v);
        }
        assert_eq!(fast, reference);
        let mut reader = fast.reader();
        let mut skipper = fast.reader();
        for _ in 0..lead {
            reader.read_bit().expect("lead bit");
            skipper.read_bit().expect("lead bit");
        }
        for &v in &values {
            assert_eq!(decode_elias_delta(&mut reader).expect("decodes"), v);
            skip_elias_delta(&mut skipper).expect("skips");
            assert_eq!(reader.position(), skipper.position());
        }
        assert_eq!(reader.remaining_bits(), 0);
    }
    // A truncated codeword is refused by both, at every cut.
    let mut one = BitString::new();
    encode_elias_delta(&mut one, (1u64 << 40) + 12345).expect("codeword");
    for cut in 0..one.len_bits() {
        let mut reader = one.reader();
        let mut part = reader.bounded_subreader(cut).expect("prefix");
        assert!(decode_elias_delta(&mut part.clone()).is_err());
        assert!(skip_elias_delta(&mut part).is_err());
    }
}

/// Indices of three and a half chunks with nondecreasing row leads every 1000 reals.
fn chunked_indices() -> (Vec<i64>, Vec<usize>) {
    let n = 3 * LATTICE_CHUNK + LATTICE_CHUNK / 2 + 17;
    let indices: Vec<i64> = (0..n)
        .map(|k| if k % 1000 == 0 { (k / 1000) as i64 * 3 } else { (((k * 7919) % 200_003) as i64 - 100_000) * if k % 3 == 0 { 1 << 20 } else { 1 } })
        .collect();
    let leads = (0..n).step_by(1000).collect();
    (indices, leads)
}

#[test]
fn chunked_lattices_are_the_sequential_message_and_read_back_identically() {
    let precision = DeclaredPrecision::new(12).expect("precision");
    let (indices, leads) = chunked_indices();
    for leads in [Vec::new(), leads] {
        let mut chunked = BitString::new();
        write_lattice_indices(&mut chunked, &indices, precision, &leads).expect("chunked");
        let mut sequential = BitString::new();
        encode_prefix_integer(&mut sequential, indices.len() as u64 + 1).expect("count");
        encode_signed_prefix_integer(&mut sequential, i64::from(precision.fraction_bits())).expect("precision");
        write_index_range(&mut sequential, &indices, &leads, 0..indices.len()).expect("one pass");
        assert_eq!(chunked, sequential);
        assert_eq!(chunked.len_bits(), ordered_lattice_bits(&indices, precision, &leads).expect("length"));
        let (expected_precision, expected) = read_lattice(&mut chunked.reader(), &leads).expect("sequential read");
        let mut skipper = chunked.reader();
        let starts = skip_lattice(&mut skipper).expect("skip");
        assert_eq!(skipper.remaining_bits(), 0);
        assert_eq!(starts.len(), indices.len().div_ceil(LATTICE_CHUNK));
        let mut reader = chunked.reader();
        let (read_precision, values) = read_lattice_chunked(&mut reader, &leads, &starts).expect("chunked read");
        assert_eq!(reader.remaining_bits(), 0);
        assert_eq!(read_precision, expected_precision);
        assert_eq!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        // Wrong chunk starts are refused, never silently misread.
        let mut shifted = starts.clone();
        shifted[1] += 1;
        assert!(read_lattice_chunked(&mut chunked.reader(), &leads, &shifted).is_err());
    }
}

/// A program of large dense operators (ordered and plain rows) around small ones.
fn large_program() -> OperatorProgram {
    let p = DeclaredPrecision::new(16).expect("precision");
    let wide = Interface::native(1100).expect("interface");
    let narrow = Interface::native(700).expect("interface");
    let ordered = Array2::from_shape_fn((700, 1100), |(i, j)| if j == 0 { i as f64 / 64.0 } else { ((i * 31 + j * 17) as f64 * 0.37).sin() });
    let plain = Array2::from_shape_fn((1100, 700), |(i, j)| ((i * 13 + j * 7) as f64 * 0.731).cos() * 3.0);
    let dense = |name: &str, rows: &Interface, cols: &Interface, values: Array2<f64>| {
        Arc::new(Operator::dense(name, rows.clone(), cols.clone(), values, p, Provenance::native(name)).expect("dense"))
    };
    OperatorProgram {
        declarations: Declarations { domains: vec![], parameters: 0, slots: vec![Slot::Raw { width: 1100 }] },
        bases: vec![],
        rules: vec![],
        operators: vec![
            dense("read", &narrow, &wide, ordered),
            Arc::new(Operator::identity("id", narrow.clone())),
            dense("write", &wide, &narrow, plain),
            dense("small", &narrow, &narrow, Array2::from_shape_fn((700, 700), |(i, j)| if i == j { 0.5 } else { 0.0 })),
        ],
        nodes: vec![
            Node::Raw { slot: 0 },
            Node::Affine { terms: vec![(0, 0)], bias: None },
            Node::Affine { terms: vec![(1, 1), (1, 3)], bias: None },
            Node::Affine { terms: vec![(2, 2)], bias: None },
        ],
        output: 3,
    }
}

#[test]
fn parallel_program_codec_matches_the_sequential_operator_codewords() {
    let program = large_program();
    let message = program.encode().expect("encode");
    assert_eq!(message.len_bits(), program.code_bits().expect("length"));
    // The operator codewords, written one at a time, are the message's operator section.
    let mut reader = message.reader();
    decode_prefix_integer(&mut reader).expect("bases");
    decode_prefix_integer(&mut reader).expect("operators");
    decode_prefix_integer(&mut reader).expect("nodes");
    for (index, operator) in program.operators.iter().enumerate() {
        let mut one = BitString::new();
        encode_operator(&mut one, operator).expect("operator codeword");
        let start = reader.position();
        let mut sequential = reader.clone();
        let expected = decode_operator(&mut sequential, index as u64).expect("sequential decode");
        let mut skipper = reader.clone();
        let starts = skip_operator(&mut skipper, index as u64).expect("skip");
        assert_eq!(skipper.position(), sequential.position());
        assert_eq!(sequential.position() - start, one.len_bits());
        assert!(reader.consume_exact_prefix(&one));
        let mut window = message.reader().window(start, reader.position()).expect("window");
        let chunked = decode_operator_with(&mut window, index as u64, Some(&starts)).expect("chunked decode");
        assert_eq!(window.remaining_bits(), 0);
        assert_eq!(*chunked, *expected);
    }
    let decoded = OperatorProgram::decode(&message, &program.declarations).expect("decode");
    for (index, (a, b)) in decoded.operators.iter().zip(&program.operators).enumerate() {
        assert_eq!(a.name, format!("decoded{index}"));
        assert_eq!((&a.rows, &a.cols, &a.body), (&b.rows, &b.cols, &b.body));
    }
    assert_eq!(decoded.encode().expect("re-encode"), message);
}

#[test]
fn moved_native_codewords_decode_to_the_ordinary_operators_and_reencode_by_content() {
    let raw = large_program();
    let source = OperatorProgram::decode(&raw.encode().expect("encode"), &raw.declarations).expect("decode");
    let cache = NativeOperatorCodec::new(&source, usize::MAX).expect("cache");
    // Drop the first operator: every later codeword moves one index down.
    let mut moved = source.clone();
    moved.operators.remove(0);
    moved.operators.push(Arc::new(Operator::identity("new", Interface::native(1100).expect("interface"))));
    // The moved operators stay in the message though no node reads them.
    moved.nodes = vec![Node::Raw { slot: 0 }, Node::Affine { terms: vec![(0, 3)], bias: None }];
    moved.output = 1;
    let message = moved.encode().expect("ordinary encode");
    assert_eq!(moved.encode_with_native_codec(&cache).expect("cached encode"), message);
    let ordinary = OperatorProgram::decode(&message, &moved.declarations).expect("ordinary decode");
    let before = cache.usage();
    let cached = OperatorProgram::decode_with_native_codec(&message, &moved.declarations, &cache).expect("cached decode");
    assert!(cache.usage().decoded_native_operator_hits >= before.decoded_native_operator_hits + 3);
    assert_eq!(cached.operators.len(), ordinary.operators.len());
    for (a, b) in cached.operators.iter().zip(&ordinary.operators) {
        assert_eq!(**a, **b);
    }
    let before = cache.usage();
    assert_eq!(cached.encode_with_native_codec(&cache).expect("content hits"), message);
    assert!(cache.usage().encoded_native_operator_hits >= before.encoded_native_operator_hits + 3);
}
