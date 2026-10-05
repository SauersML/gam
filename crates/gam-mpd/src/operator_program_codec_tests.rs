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

