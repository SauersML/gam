//! Declared-precision real codes and decode-then-evaluate distortion (#2951).
//!
//! A mechanism program's code carries no free real-valued coefficient. Every real
//! an artifact holds is sent as integers under a precision the experiment declares,
//! and the decoder rebuilds it from those integers and the declaration alone.
//! Fidelity is then measured on what the decoder rebuilt.
//!
//! # Dyadic lattice codes
//!
//! A [`DeclaredPrecision`] is the step `2^-p` for a declared integer `p`, so the
//! precision is itself an integer and costs integer bits wherever an artifact
//! transmits it. A real `x` is sent as `k = round(x·2^p)` and decoded as
//! `x̂ = k·2^-p`. Scaling by a power of two is exact in binary floating point and
//! rounding to an integer is exact, so for `|k| ≤ 2^53` the index and the decoded
//! value are both exact and
//!
//! ```text
//!   |x − x̂| ≤ 2^-(p+1)
//! ```
//!
//! holds with no rounding term. A step that is not a power of two would put a
//! rounding error into every decoded value.
//!
//! # Decode, then evaluate
//!
//! [`decode_then_evaluate`] decodes an artifact, executes the decoded artifact, and
//! measures the declared distortion of its outputs against the native reference.
//! The quantization error of the parameters never stands in for the error of the
//! outputs: the computation is nonlinear, and a parameter error bounds an output
//! error only through a derived Lipschitz constant. The measurement covers the inputs
//! the evaluation executes and certifies nothing beyond them (#2946 fr-census
//! overclaim audit, comment 5716123817).
//!
//! The declared distortion returns an [`EvidenceStatus`]. The status names the family
//! it evaluated, its witness and its derived rounding bound.
//! [`DecodedFidelity::verdict`] reports only what that status proves about the
//! tolerance. A figure within its rounding of the tolerance is unresolved, not a pass.
//! A statistical estimate never certifies.
//!
//! # Messages
//!
//! The code is also one self-delimiting message through `codec`'s integer codes, so
//! a library appends it and a decoder reads it back without a real-valued field:
//! [`LatticeCode::write`] sends the coordinate count as `count + 1` in the prefix
//! integer code, then the declared precision `p` and each index `k` in `codec`'s signed
//! prefix integer code. [`LatticeCode::index_bits`] is the length of the index codewords
//! alone, without that header.
//!
//! A reader refuses a count that the bits remaining in the message cannot hold before
//! it allocates anything: a lattice index takes at least one bit.

/// Largest `|p|` for which both `2^p` and `2^-p` are normal `f64` powers of two: the
/// normal exponents span `[f64::MIN_EXP - 1, f64::MAX_EXP - 1] = [-1022, 1023]`.
const EXPONENT_LIMIT: i32 = -(f64::MIN_EXP - 1);

/// Every integer of magnitude at most `2^53` is an `f64`, so an index in this range
/// decodes exactly.
const INDEX_LIMIT: u64 = 1 << f64::MANTISSA_DIGITS;

/// `2^exponent` for a normal exponent in `[-1022, 1023]`, built from its bits: an
/// all-zero fraction under the biased exponent `exponent + 1023`.
fn power_of_two(exponent: i32) -> f64 {
    f64::from_bits(((exponent + f64::MAX_EXP - 1) as u64) << (f64::MANTISSA_DIGITS - 1))
}

/// A declared precision for real coordinates: the dyadic step `2^-fraction_bits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeclaredPrecision {
    fraction_bits: i32,
}

impl DeclaredPrecision {
    /// The step `2^-fraction_bits`: binary digits after the point, negative for a
    /// step above one. Refused when `2^fraction_bits` or `2^-fraction_bits` is not a
    /// normal `f64`, because the encoder and decoder then stop being exact scalings.
    pub fn new(fraction_bits: i32) -> Result<Self, String> {
        if fraction_bits.unsigned_abs() > EXPONENT_LIMIT.unsigned_abs() {
            return Err(format!(
                "DeclaredPrecision: fraction bits {fraction_bits} put the step 2^-{fraction_bits} \
                 or its inverse outside the normal f64 exponents (|p| <= {EXPONENT_LIMIT})"
            ));
        }
        Ok(Self { fraction_bits })
    }

    /// The declared binary digits after the point.
    pub fn fraction_bits(self) -> i32 {
        self.fraction_bits
    }

    /// The lattice step `2^-fraction_bits`.
    pub fn step(self) -> f64 {
        power_of_two(-self.fraction_bits)
    }

    /// Round one value exactly as encoding and decoding a singleton lattice code,
    /// without allocating an index or decoded-value vector. Ties round away from
    /// zero; the integer representative canonicalizes either signed zero to +0.
    /// Refusals have the same messages as the singleton code path.
    pub fn round(self, value: f64) -> Result<f64, String> {
        if !value.is_finite() {
            return Err(format!(
                "LatticeCode::encode: value {value} at position 0 is not finite"
            ));
        }
        let index = (value * power_of_two(self.fraction_bits)).round();
        if !(index.abs() <= INDEX_LIMIT as f64) {
            return Err(format!(
                "LatticeCode::encode: value {value} at position 0 needs index {index} \
                 at step 2^-{}, beyond the exactly decodable 2^53",
                self.fraction_bits
            ));
        }
        let integer_index = index as i64;
        let decoded = integer_index as f64 * self.step();
        if !decoded.is_finite() {
            return Err(format!(
                "LatticeCode: index {integer_index} at position 0 overflows at step 2^-{}",
                self.fraction_bits
            ));
        }
        Ok(decoded)
    }

    /// The finest precision for reals of magnitude at most `largest`: this one, coarsened
    /// to `2^-(53 − ⌈log₂ largest⌉)` when that is coarser, so every index stays within
    /// `2^53` and decodes exactly. The step is derived from the value range; a precision
    /// chosen for other reals (an operator's former values, a curvature step) never
    /// overflows the lattice code. A zero or non-finite `largest` leaves it unchanged (a
    /// non-finite real is refused by the encoder).
    pub fn within_range(self, largest: f64) -> Self {
        if !(largest > 0.0 && largest.is_finite()) {
            return self;
        }
        // Extract the exact binary exponent: log2 can round a value immediately
        // above a power of two back to that integer. Every index through 2^53
        // is representable, so retaining 53 (not 52) bits is required to avoid
        // changing an otherwise exactly encodable binary64 literal.
        let bits = largest.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        let ceil_log2 = if exponent == 0 {
            -1074 + fraction.ilog2() as i32 + i32::from(!fraction.is_power_of_two())
        } else {
            exponent - 1023 + i32::from(fraction != 0)
        };
        let cap = 53 - ceil_log2;
        Self { fraction_bits: self.fraction_bits.min(cap.max(-EXPONENT_LIMIT)) }
    }

}

/// An artifact that a decoder rebuilds from integers and declarations alone.
pub trait DecodableArtifact {
    /// What the decoder rebuilds.
    type Decoded;

    /// Rebuild the artifact from its transmitted integers.
    fn decode(&self) -> Result<Self::Decoded, String>;
}

/// Reals sent as indices on the dyadic lattice of one [`DeclaredPrecision`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticeCode {
    precision: DeclaredPrecision,
    indices: Vec<i64>,
}

impl LatticeCode {
    /// Encode each value as its nearest lattice index `round(x·2^p)`, with ties rounded
    /// away from zero. Refused for a non-finite value, for an index beyond `2^53`, or
    /// when the decoded value would overflow.
    pub fn encode(values: &[f64], precision: DeclaredPrecision) -> Result<Self, String> {
        let scale = power_of_two(precision.fraction_bits);
        let mut indices = Vec::with_capacity(values.len());
        for (position, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "LatticeCode::encode: value {value} at position {position} is not finite"
                ));
            }
            let index = (value * scale).round();
            if !(index.abs() <= INDEX_LIMIT as f64) {
                return Err(format!(
                    "LatticeCode::encode: value {value} at position {position} needs index {index} \
                     at step 2^-{}, beyond the exactly decodable 2^53",
                    precision.fraction_bits
                ));
            }
            indices.push(index as i64);
        }
        Self::from_indices(precision, indices)
    }

    /// Accept transmitted indices. Refused for an index beyond `2^53` or one whose
    /// decoded value overflows.
    pub fn from_indices(precision: DeclaredPrecision, indices: Vec<i64>) -> Result<Self, String> {
        let step = precision.step();
        for (position, &index) in indices.iter().enumerate() {
            if index.unsigned_abs() > INDEX_LIMIT {
                return Err(format!(
                    "LatticeCode: index {index} at position {position} is beyond the exactly \
                     decodable 2^53"
                ));
            }
            if !(index as f64 * step).is_finite() {
                return Err(format!(
                    "LatticeCode: index {index} at position {position} overflows at step 2^-{}",
                    precision.fraction_bits
                ));
            }
        }
        Ok(Self { precision, indices })
    }

    /// The transmitted lattice indices.
    pub fn indices(&self) -> &[i64] {
        &self.indices
    }

    /// The transmitted lattice indices, owned.
    pub fn into_indices(self) -> Vec<i64> {
        self.indices
    }
}

impl DecodableArtifact for LatticeCode {
    type Decoded = Vec<f64>;

    fn decode(&self) -> Result<Vec<f64>, String> {
        let step = self.precision.step();
        Ok(self.indices.iter().map(|&index| index as f64 * step).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    

    fn assert_scalar_matches_code(value: f64, precision: DeclaredPrecision) {
        let reference = LatticeCode::encode(&[value], precision)
            .and_then(|code| code.decode())
            .map(|decoded| decoded[0].to_bits());
        assert_eq!(
            precision.round(value).map(f64::to_bits),
            reference,
            "value bits {:016x}, precision {}",
            value.to_bits(),
            precision.fraction_bits()
        );
    }

    #[test]
    fn scalar_round_matches_code_at_every_declared_exponent() {
        for exponent in -EXPONENT_LIMIT..=EXPONENT_LIMIT {
            let precision = DeclaredPrecision::new(exponent).unwrap();
            let step = precision.step();
            for value in [
                0.,
                -0.,
                f64::from_bits(1),
                -f64::from_bits(1),
                f64::MIN_POSITIVE,
                -f64::MIN_POSITIVE,
                f64::MAX,
                -f64::MAX,
                0.5 * step,
                -0.5 * step,
                1.5 * step,
                -1.5 * step,
                INDEX_LIMIT as f64 * step,
                -(INDEX_LIMIT as f64) * step,
                f64::from_bits((INDEX_LIMIT as f64).to_bits() + 1) * step,
            ] {
                assert_scalar_matches_code(value, precision);
            }
        }
    }

    #[test]
    fn scalar_round_preserves_ties_zero_and_refusals() {
        let precision = DeclaredPrecision::new(0).unwrap();
        for (value, expected) in [(0.5, 1_f64), (-0.5, -1.), (1.5, 2.), (-1.5, -2.)] {
            assert_eq!(
                precision.round(value).unwrap().to_bits(),
                expected.to_bits()
            );
        }
        for value in [0., -0., f64::from_bits(1), -f64::from_bits(1)] {
            assert_eq!(precision.round(value).unwrap().to_bits(), 0_f64.to_bits());
        }
        for value in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits((INDEX_LIMIT as f64).to_bits() + 1),
        ] {
            assert_scalar_matches_code(value, precision);
            assert!(precision.round(value).is_err());
        }
        // This finite input rounds to an index whose decoded value overflows.
        let coarse = DeclaredPrecision::new(-EXPONENT_LIMIT).unwrap();
        assert_scalar_matches_code(f64::MAX, coarse);
        assert!(coarse.round(f64::MAX).unwrap_err().contains("overflows"));
    }

    #[test]
    fn scalar_round_matches_code_on_seeded_float_bit_patterns() {
        let mut state = 0x243f_6a88_85a3_08d3_u64;
        for exponent in -EXPONENT_LIMIT..=EXPONENT_LIMIT {
            let precision = DeclaredPrecision::new(exponent).unwrap();
            for sample in 0..32 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let value = f64::from_bits(state);
                assert_scalar_matches_code(value, precision);
                if sample % 2 == 0 {
                    assert_scalar_matches_code(-value, precision);
                }
            }
        }
    }

    #[test]
    fn lattice_code_refuses_what_it_cannot_decode_exactly() {
        assert!(DeclaredPrecision::new(EXPONENT_LIMIT).is_ok());
        assert!(DeclaredPrecision::new(-EXPONENT_LIMIT).is_ok());
        assert!(DeclaredPrecision::new(EXPONENT_LIMIT + 1).is_err());
        assert!(DeclaredPrecision::new(i32::MIN).is_err());

        let unit = DeclaredPrecision::new(0).expect("unit step");
        assert!(LatticeCode::encode(&[f64::NAN], unit).is_err());
        assert!(LatticeCode::encode(&[f64::NEG_INFINITY], unit).is_err());
        // Index guard: 2^53 is accepted, 2^54 is refused.
        assert!(LatticeCode::encode(&[INDEX_LIMIT as f64], unit).is_ok());
        assert!(LatticeCode::encode(&[2.0 * INDEX_LIMIT as f64], unit).is_err());
        assert!(LatticeCode::from_indices(unit, vec![INDEX_LIMIT as i64]).is_ok());
        assert!(LatticeCode::from_indices(unit, vec![-(INDEX_LIMIT as i64) - 1]).is_err());
        // Overflow guard at the coarsest step 2^1022: 2^1023 decodes, while f64::MAX
        // rounds to index 4, whose decoded 2^1024 overflows.
        let coarsest = DeclaredPrecision::new(-EXPONENT_LIMIT).expect("coarsest step");
        assert!(LatticeCode::encode(&[power_of_two(f64::MAX_EXP - 1)], coarsest).is_ok());
        assert!(LatticeCode::encode(&[f64::MAX], coarsest).is_err());
    }

    

}

#[cfg(test)]
mod exact_range_tests {
    use super::DeclaredPrecision;

    #[test]
    fn singleton_binary64_literals_keep_all_significand_bits() {
        for value in [1e-5_f64, 0.1, std::f64::consts::PI, 1.0_f64.next_up(), 2.0_f64.next_down(), 2.0_f64.next_up()] {
            let precision = DeclaredPrecision::new(100).unwrap().within_range(value);
            assert_eq!(precision.round(value).unwrap(), value, "value {value}, precision {precision:?}");
        }
    }

    #[test]
    fn exponent_boundary_is_exact_and_indices_stay_in_range() {
        for exponent in -100..100 {
            let power = f64::from_bits(((exponent + 1023) as u64) << 52);
            for value in [power.next_down(), power, power.next_up()] {
                let precision = DeclaredPrecision::new(1000).unwrap().within_range(value);
                assert_eq!(precision.round(value).unwrap(), value);
            }
        }
        for value in [f64::MIN_POSITIVE, f64::from_bits(1), f64::MAX] {
            DeclaredPrecision::new(1022).unwrap().within_range(value).round(value).unwrap();
        }
    }
}

