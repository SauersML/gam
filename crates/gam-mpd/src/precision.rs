//! Declared-precision real codes, quotient representatives and decode-then-evaluate
//! distortion (#2951, P11 and P18).
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
//! # Quotient representatives
//!
//! A coordinate of `R/TZ`, such as the projective line `RP1` as `t ~ t + π` (P11) or
//! a turn angle with period `2π`, names one class. A code that sends `t` and `t + T`
//! differently spends bits on the gauge. [`QuotientCode`] sends the class once: the
//! canonical representative in `[0, T)` is a convention the encoder and decoder
//! share, and the transmitted index names the nearest of `N = 2^b` cells of the
//! period, at `b` bits per coordinate.
//!
//! With `u` the unit roundoff, the encoder computes:
//! - `r = t mod T`. The floating-point remainder is exact; only the correction
//!   `r + T` of a negative remainder rounds, by at most `uT`.
//! - The fraction `s = r/T`, one division, rounding by at most `u` since `r ≤ T`.
//! - `j = round(s·N) mod N`, which is exact.
//!
//! The decoder returns `j·(T/N)`. The cell width is an exact scaling, and the
//! product rounds by at most `uT`. In period units the decoded representative is
//! within `1/(2N) + 3u` of the input's class, so
//!
//! ```text
//!   d_{R/TZ}(t, t̂) < T·(2^-(b+1) + 3u).
//! ```
//!
//! [`QuotientCode::worst_case_error`] reports `T·2^-(b+1) + 4uT`. Both terms are exact
//! power-of-two scalings of `T`, and the extra `uT` absorbs the one rounded addition,
//! so the figure never falls below the bound. A cell whose half-width `2^-(b+1)` does
//! not exceed the `3u` rounding floor is not resolved and is refused.
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
//! Each code is also one self-delimiting message through `codec`'s integer codes, so
//! a library appends it and a decoder reads it back without a real-valued field:
//!
//! * [`LatticeCode::write`] sends the coordinate count as `count + 1` in the prefix
//!   integer code, then the declared precision `p` and each index `k` in `codec`'s
//!   signed prefix integer code.
//!   [`LatticeCode::index_bits`] is the length of the index codewords alone, without
//!   that header.
//! * [`QuotientCode::write`] sends the count as `count + 1` and the resolution as
//!   `b + 1` in the prefix integer code, then each index as a fixed index into the
//!   `2^b` cells: exactly [`QuotientCode::index_bits`] after the header. The period
//!   is a declared convention of the decoder and is never written.
//!
//! A reader refuses a count that the bits remaining in the message cannot hold before
//! it allocates anything: a lattice index takes at least one bit, and a quotient index
//! exactly `b`. That bound needs `b ≥ 1`. A 0-bit quotient code would decode every class
//! to the same representative and carry no coordinate, so it is refused.

use super::codec::{
    BitReader, BitString, decode_fixed_index, decode_prefix_integer, decode_signed_prefix_integer,
    encode_fixed_index, encode_prefix_integer, encode_signed_prefix_integer,
    signed_prefix_integer_len_bits,
};
use super::supports::EvidenceStatus;
use gam_linalg::roundoff::UNIT_ROUNDOFF;

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

    /// The exact worst-case decoding error `2^-(fraction_bits+1)`, with no rounding
    /// term (see the module note).
    pub fn worst_case_error(self) -> f64 {
        0.5 * self.step()
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

    /// The declared precision.
    pub fn precision(&self) -> DeclaredPrecision {
        self.precision
    }

    /// The transmitted lattice indices.
    pub fn indices(&self) -> &[i64] {
        &self.indices
    }
}

impl DecodableArtifact for LatticeCode {
    type Decoded = Vec<f64>;

    fn decode(&self) -> Result<Vec<f64>, String> {
        let step = self.precision.step();
        Ok(self.indices.iter().map(|&index| index as f64 * step).collect())
    }
}

/// The quotient `R/TZ` of the real line by a period `T`, such as `RP1` with `T = π`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicQuotient {
    period: f64,
}

impl PeriodicQuotient {
    /// The quotient by `period`. Refused unless the period is finite and positive, and
    /// its scaling by `2^-53` is still normal. That condition makes every power-of-two
    /// scaling of the period used here (at most `2^-51`) exact.
    pub fn new(period: f64) -> Result<Self, String> {
        let admissible = period.is_finite()
            && period > 0.0
            && (period * power_of_two(-(f64::MANTISSA_DIGITS as i32))).is_normal();
        if !admissible {
            return Err(format!(
                "PeriodicQuotient: period {period} must be finite and positive, with its 2^-53 \
                 scaling still a normal f64"
            ));
        }
        Ok(Self { period })
    }

    /// The period `T`.
    pub fn period(self) -> f64 {
        self.period
    }
}

/// Refuse a resolution with no bits, or one whose cell half-width `2^-(b+1)`, in period
/// units, does not exceed the `3u` rounding of the representative (see the module note).
/// A 0-bit code decodes every class to one representative and carries no coordinate.
fn check_resolution(resolution_bits: u32) -> Result<(), String> {
    if resolution_bits == 0 {
        return Err(
            "QuotientCode: a 0-bit code carries no coordinate: every class decodes to 0"
                .to_string(),
        );
    }
    let resolved = resolution_bits < f64::MANTISSA_DIGITS
        && power_of_two(-(resolution_bits as i32) - 1) > 3.0 * UNIT_ROUNDOFF;
    if !resolved {
        return Err(format!(
            "QuotientCode: {resolution_bits} resolution bits make cells no wider than the \
             rounding of their representative"
        ));
    }
    Ok(())
}

/// Coordinates of a [`PeriodicQuotient`] sent once per class, as cell indices of the
/// canonical representative in `[0, T)`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuotientCode {
    quotient: PeriodicQuotient,
    resolution_bits: u32,
    indices: Vec<u64>,
}

impl QuotientCode {
    /// Encode each value's class as the nearest of `2^resolution_bits` cells of the
    /// period. Refused for a non-finite value or an unresolved resolution.
    pub fn encode(
        values: &[f64],
        quotient: PeriodicQuotient,
        resolution_bits: u32,
    ) -> Result<Self, String> {
        check_resolution(resolution_bits)?;
        let cells = power_of_two(resolution_bits as i32);
        let modulus = 1_u64 << resolution_bits;
        let period = quotient.period;
        let mut indices = Vec::with_capacity(values.len());
        for (position, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "QuotientCode::encode: value {value} at position {position} is not finite"
                ));
            }
            let fraction = value.rem_euclid(period) / period;
            indices.push((fraction * cells).round() as u64 % modulus);
        }
        Ok(Self {
            quotient,
            resolution_bits,
            indices,
        })
    }

    /// Accept transmitted cell indices. Refused for an unresolved resolution or an
    /// index outside `[0, 2^resolution_bits)`.
    pub fn from_indices(
        quotient: PeriodicQuotient,
        resolution_bits: u32,
        indices: Vec<u64>,
    ) -> Result<Self, String> {
        check_resolution(resolution_bits)?;
        let modulus = 1_u64 << resolution_bits;
        if let Some(position) = indices.iter().position(|&index| index >= modulus) {
            return Err(format!(
                "QuotientCode: index {} at position {position} is outside [0, 2^{resolution_bits})",
                indices[position]
            ));
        }
        Ok(Self {
            quotient,
            resolution_bits,
            indices,
        })
    }

    /// The quotient the coordinates live in.
    pub fn quotient(&self) -> PeriodicQuotient {
        self.quotient
    }

    /// Bits per coordinate.
    pub fn resolution_bits(&self) -> u32 {
        self.resolution_bits
    }

    /// The transmitted cell indices.
    pub fn indices(&self) -> &[u64] {
        &self.indices
    }

    /// The exact length of the fixed-rate index code: `resolution_bits` per coordinate.
    pub fn index_bits(&self) -> u64 {
        u64::from(self.resolution_bits) * self.indices.len() as u64
    }

    /// An upper bound on the quotient distance between each input's class and its
    /// decoded representative, rounding included: `T·2^-(b+1) + 4uT` (see the module
    /// note).
    pub fn worst_case_error(&self) -> f64 {
        let period = self.quotient.period;
        period * power_of_two(-(self.resolution_bits as i32) - 1) + 4.0 * UNIT_ROUNDOFF * period
    }
}

impl DecodableArtifact for QuotientCode {
    type Decoded = Vec<f64>;

    fn decode(&self) -> Result<Vec<f64>, String> {
        let cell_width = self.quotient.period * power_of_two(-(self.resolution_bits as i32));
        Ok(self
            .indices
            .iter()
            .map(|&index| index as f64 * cell_width)
            .collect())
    }
}

impl LatticeCode {
    /// Append the code as one message. The count goes as `count + 1` in the prefix integer
    /// code; the declared precision and each index go in the signed prefix integer code
    /// (see the module note).
    pub fn write(&self, out: &mut BitString) -> Result<(), String> {
        encode_prefix_integer(out, self.indices.len() as u64 + 1)
            .map_err(|error| format!("LatticeCode::write: count: {error}"))?;
        encode_signed_prefix_integer(out, i64::from(self.precision.fraction_bits))
            .map_err(|error| format!("LatticeCode::write: precision: {error}"))?;
        for (position, &index) in self.indices.iter().enumerate() {
            encode_signed_prefix_integer(out, index)
                .map_err(|error| format!("LatticeCode::write: index {position}: {error}"))?;
        }
        Ok(())
    }

    /// Read one message written by [`Self::write`]. Every index codeword takes at least
    /// one bit, so a count above the bits that remain is refused before anything is
    /// allocated. The indices then pass [`Self::from_indices`].
    pub fn read(reader: &mut BitReader<'_>) -> Result<Self, String> {
        let count = decode_prefix_integer(reader)
            .map_err(|error| format!("LatticeCode::read: count: {error}"))?
            - 1;
        let exponent = decode_signed_prefix_integer(reader)
            .map_err(|error| format!("LatticeCode::read: precision: {error}"))?;
        let fraction_bits = i32::try_from(exponent).map_err(|error| {
            format!("LatticeCode::read: precision exponent {exponent}: {error}")
        })?;
        let precision = DeclaredPrecision::new(fraction_bits)?;
        let remaining = reader.remaining_bits();
        if count > remaining {
            return Err(format!(
                "LatticeCode::read: {count} indices need at least {count} bits, {remaining} remain"
            ));
        }
        let mut indices = Vec::with_capacity(count as usize);
        for position in 0..count {
            let index = decode_signed_prefix_integer(reader)
                .map_err(|error| format!("LatticeCode::read: index {position}: {error}"))?;
            indices.push(index);
        }
        Self::from_indices(precision, indices)
    }

    /// The exact length of the index codewords in the signed prefix integer code. It counts
    /// the indices only, not the count and precision header that [`Self::write`] adds,
    /// mirroring [`QuotientCode::index_bits`]. A written message is its header plus this
    /// many bits.
    pub fn index_bits(&self) -> Result<u64, String> {
        let mut bits = 0_u64;
        for (position, &index) in self.indices.iter().enumerate() {
            bits += signed_prefix_integer_len_bits(index)
                .map_err(|error| format!("LatticeCode::index_bits: index {position}: {error}"))?;
        }
        Ok(bits)
    }
}

impl QuotientCode {
    /// Append the code as one message: the count as `count + 1` and the resolution as
    /// `b + 1` in the prefix integer code, then each index as a fixed index into the
    /// `2^b` cells (see the module note). The period is not written.
    pub fn write(&self, out: &mut BitString) -> Result<(), String> {
        encode_prefix_integer(out, self.indices.len() as u64 + 1)
            .map_err(|error| format!("QuotientCode::write: count: {error}"))?;
        encode_prefix_integer(out, u64::from(self.resolution_bits) + 1)
            .map_err(|error| format!("QuotientCode::write: resolution: {error}"))?;
        let cells = 1_usize << self.resolution_bits;
        for (position, &index) in self.indices.iter().enumerate() {
            encode_fixed_index(out, index as usize, cells)
                .map_err(|error| format!("QuotientCode::write: index {position}: {error}"))?;
        }
        Ok(())
    }

    /// Read one message written by [`Self::write`] for the declared `quotient`. Each index
    /// takes exactly `b >= 1` bits, so a count above the remaining bits over `b` is refused
    /// before anything is allocated. The indices then pass [`Self::from_indices`].
    pub fn read(reader: &mut BitReader<'_>, quotient: PeriodicQuotient) -> Result<Self, String> {
        let count = decode_prefix_integer(reader)
            .map_err(|error| format!("QuotientCode::read: count: {error}"))?
            - 1;
        let resolution = decode_prefix_integer(reader)
            .map_err(|error| format!("QuotientCode::read: resolution: {error}"))?
            - 1;
        let resolution_bits = u32::try_from(resolution)
            .map_err(|error| format!("QuotientCode::read: resolution {resolution}: {error}"))?;
        check_resolution(resolution_bits)?;
        let remaining = reader.remaining_bits();
        if count > remaining / u64::from(resolution_bits) {
            return Err(format!(
                "QuotientCode::read: {count} indices of {resolution_bits} bits exceed the \
                 {remaining} bits that remain"
            ));
        }
        let cells = 1_usize << resolution_bits;
        let mut indices = Vec::with_capacity(count as usize);
        for position in 0..count {
            let index = decode_fixed_index(reader, cells)
                .map_err(|error| format!("QuotientCode::read: index {position}: {error}"))?;
            indices.push(index as u64);
        }
        Self::from_indices(quotient, resolution_bits, indices)
    }
}

/// What a decoded artifact's distortion evidence proves about the declared tolerance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FidelityVerdict {
    /// The status proves the distortion is at most the tolerance.
    Meets,
    /// The status proves the distortion exceeds the tolerance.
    Violates,
    /// Neither is proven. Either the tolerance lies inside the figure's rounding band, or
    /// the status proves no bound on that side (a statistical estimate proves none).
    Unresolved,
}

/// The evidence a decoded artifact's declared distortion carries, under the declared
/// fidelity tolerance.
///
/// The status is stated over the inputs the evaluation executed and bounds nothing
/// beyond them (#2946 fr-census overclaim audit, comment 5716123817).
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedFidelity<W, D> {
    status: EvidenceStatus<W, D>,
    tolerance: f64,
}

impl<W, D> DecodedFidelity<W, D> {
    /// The evidence the distortion measurement carries.
    pub fn status(&self) -> &EvidenceStatus<W, D> {
        &self.status
    }

    /// The declared fidelity tolerance.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// What the status proves about the tolerance. The owner of the rounding is
    /// [`EvidenceStatus`]: an exact figure with no numerical error is never unresolved.
    pub fn verdict(&self) -> FidelityVerdict {
        if self.status.certifies_at_most(self.tolerance) {
            FidelityVerdict::Meets
        } else if self.status.refutes_at_most(self.tolerance) {
            FidelityVerdict::Violates
        } else {
            FidelityVerdict::Unresolved
        }
    }
}

/// Decode `artifact`, execute the decoded artifact with `evaluate`, and state the
/// declared distortion of its outputs against the native reference as evidence under
/// the declared `tolerance`.
///
/// The evaluation receives only what the decoder rebuilt. `distortion` names what it
/// evaluated (its family, witness and domain) and its derived rounding bound through
/// the [`EvidenceStatus`] it returns. The tolerance is a required experiment
/// declaration and is refused unless finite and non-negative.
pub fn decode_then_evaluate<A, O, E, M, W, D>(
    artifact: &A,
    evaluate: E,
    native_reference: &O,
    distortion: M,
    tolerance: f64,
) -> Result<DecodedFidelity<W, D>, String>
where
    A: DecodableArtifact,
    E: FnOnce(&A::Decoded) -> Result<O, String>,
    M: FnOnce(&O, &O) -> Result<EvidenceStatus<W, D>, String>,
{
    if !(tolerance.is_finite() && tolerance >= 0.0) {
        return Err(format!(
            "decode_then_evaluate: the declared fidelity tolerance {tolerance} must be finite \
             and non-negative"
        ));
    }
    let decoded = artifact.decode()?;
    let outputs = evaluate(&decoded)?;
    let status = distortion(&outputs, native_reference)?;
    Ok(DecodedFidelity { status, tolerance })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::f64::consts::PI;

    /// The distance between the classes of `value` and `representative` in `R/TZ`,
    /// measured to within `4uT`. The remainder `%` is exact. The subtraction rounds by
    /// at most `2uT`, since the offset is below `2T`; the Euclidean correction by at
    /// most `uT`; and the complement by at most `uT`.
    fn class_distance(value: f64, representative: f64, period: f64) -> f64 {
        let wrapped = (value % period - representative).rem_euclid(period);
        wrapped.min(period - wrapped)
    }

    /// `sum_c (2/C) v(t_c) v(t_c)^T` with `v(t) = (cos t, sin t)`, row-major.
    fn projector_sum(angles: &[f64]) -> [f64; 4] {
        let weight = 2.0 / angles.len() as f64;
        let mut sum = [0.0; 4];
        for &angle in angles {
            let (sine, cosine) = angle.sin_cos();
            sum[0] += weight * cosine * cosine;
            sum[1] += weight * cosine * sine;
            sum[2] += weight * sine * cosine;
            sum[3] += weight * sine * sine;
        }
        sum
    }

    fn frobenius_distance(left: &[f64; 4], right: &[f64; 4]) -> f64 {
        left.iter()
            .zip(right)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn lattice_decoding_error_is_at_most_half_a_step_with_no_rounding_term() {
        for fraction_bits in [3, 0, -4] {
            let precision = DeclaredPrecision::new(fraction_bits).expect("precision in range");
            let step = precision.step();
            let values = [
                0.0,
                0.5 * step,
                -0.5 * step,
                0.3 * step,
                1.0 / 3.0,
                -7.0 / 3.0,
                123.456,
                1.0e6 + 0.3,
            ];
            let code = LatticeCode::encode(&values, precision).expect("finite values encode");
            let decoded = code.decode().expect("a lattice code decodes");
            for ((&value, &index), &estimate) in values.iter().zip(code.indices()).zip(&decoded) {
                assert_eq!(estimate.to_bits(), (index as f64 * step).to_bits());
                // Either the estimate is zero or |estimate| >= step while the error is at
                // most step/2, so estimate/2 <= value <= 2·estimate (Sterbenz) and the
                // measured difference is exact: the bound is asserted with no slack.
                assert!(
                    (value - estimate).abs() <= precision.worst_case_error(),
                    "value {value} decoded to {estimate} at step {step}"
                );
            }
            // A value half a step off the lattice attains the bound, so it is tight.
            assert_eq!((values[1] - decoded[1]).abs(), precision.worst_case_error());
            // Decoded values are codewords: re-encoding them is a fixed point.
            let reencoded = LatticeCode::encode(&decoded, precision).expect("decoded values encode");
            assert_eq!(reencoded.indices(), code.indices());
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

    #[test]
    fn quotient_code_decodes_within_its_bound_of_each_class() {
        let quotient = PeriodicQuotient::new(PI).expect("the RP1 period is admissible");
        let values = [
            0.0,
            1.0,
            -1.0,
            0.5 * PI,
            PI,
            -PI,
            3.0,
            1.0e6 + 0.1,
            -12345.678,
            1.0e-300,
        ];
        let measurement_band = 4.0 * UNIT_ROUNDOFF * PI;
        for resolution_bits in [1, 7, 30, 50] {
            let code = QuotientCode::encode(&values, quotient, resolution_bits)
                .expect("finite values encode");
            let decoded = code.decode().expect("a quotient code decodes");
            let bound = code.worst_case_error();
            for (&value, &representative) in values.iter().zip(&decoded) {
                assert!(
                    (0.0..PI).contains(&representative),
                    "representative {representative} of {value} is outside [0, pi)"
                );
                let distance = class_distance(value, representative, PI);
                assert!(
                    distance <= (bound + measurement_band).next_up(),
                    "value {value} decoded to {representative} at {resolution_bits} bits: \
                     distance {distance} beyond bound {bound}"
                );
            }
            let reencoded = QuotientCode::encode(&decoded, quotient, resolution_bits)
                .expect("decoded representatives encode");
            assert_eq!(reencoded.indices(), code.indices());
            assert_eq!(
                code.index_bits(),
                u64::from(resolution_bits) * values.len() as u64
            );
        }
    }

    #[test]
    fn translates_of_a_class_share_one_quotient_index_but_not_one_lattice_index() {
        let quotient = PeriodicQuotient::new(PI).expect("the RP1 period is admissible");
        let resolution_bits = 10;
        let cell_width = PI * power_of_two(-10);
        let precision = DeclaredPrecision::new(10).expect("precision in range");
        for cell_index in [0_u64, 1, 511, 1023] {
            // A quarter cell off a codeword. Translating by up to three periods rounds by
            // at most 7u in period units: u·3pi for the product and u·4pi for the sum.
            // Encoding adds 2u. Both are far inside the quarter-cell margin of 2^-12.
            let base = (cell_index as f64 + 0.25) * cell_width;
            let translates: Vec<f64> = (-3..=3).map(|turns| base + f64::from(turns) * PI).collect();
            let code = QuotientCode::encode(&translates, quotient, resolution_bits)
                .expect("translates encode");
            assert!(
                code.indices().iter().all(|&index| index == cell_index),
                "translates of cell {cell_index} encoded to {:?}",
                code.indices()
            );
            // Negative control: the plain lattice code sends the translates as seven reals.
            let lattice = LatticeCode::encode(&translates, precision).expect("translates encode");
            let distinct: BTreeSet<i64> = lattice.indices().iter().copied().collect();
            assert_eq!(distinct.len(), translates.len());
        }
    }

    #[test]
    fn quotient_code_refuses_cells_below_the_rounding_of_their_representative() {
        let quotient = PeriodicQuotient::new(PI).expect("the RP1 period is admissible");
        // A 0-bit code carries no coordinate, while 1 bit is the smallest resolved code.
        assert!(QuotientCode::encode(&[1.0], quotient, 0).is_err());
        assert!(QuotientCode::encode(&[1.0], quotient, 1).is_ok());
        assert!(QuotientCode::from_indices(quotient, 0, vec![0]).is_err());
        // Half-width 2^-51 = 4u exceeds the 3u floor; 2^-52 = 2u does not.
        assert!(QuotientCode::encode(&[1.0], quotient, 50).is_ok());
        assert!(QuotientCode::encode(&[1.0], quotient, 51).is_err());
        assert!(QuotientCode::encode(&[1.0], quotient, u32::MAX).is_err());
        assert!(QuotientCode::encode(&[f64::NAN], quotient, 3).is_err());
        assert!(QuotientCode::from_indices(quotient, 3, vec![7]).is_ok());
        assert!(QuotientCode::from_indices(quotient, 3, vec![8]).is_err());
        assert!(PeriodicQuotient::new(1.0e-200).is_ok());
        for period in [0.0, -PI, f64::NAN, f64::INFINITY, 1.0e-300] {
            assert!(
                PeriodicQuotient::new(period).is_err(),
                "period {period} was admitted"
            );
        }
    }

    use crate::supports::ExactBasis;
    use gam_linalg::roundoff::accumulation_growth;

    /// A derived bound on the rounding in
    /// `frobenius_distance(&projector_sum(angles), &identity)` for `components` angles.
    /// - Each entry sums `components` terms `weight·x·y`. `x` and `y` are libm sine or
    ///   cosine values, at most one ulp off, so two roundings each. `weight` rounds once
    ///   and each product once: seven roundings per term, plus `components − 1` additions.
    ///   The terms have absolute sum at most `weight · components = 2`.
    /// - Subtracting the identity rounds once over magnitudes of at most 3.
    /// - The entrywise error bounds the norm's error through its Frobenius norm, at most
    ///   twice the entry bound for 4 entries. Four squares, three additions and the root
    ///   add at most `γ_8` relative to the figure.
    fn projector_distance_roundoff(components: usize, distortion: f64) -> f64 {
        let entry = accumulation_growth(components + 6) * 2.0 + accumulation_growth(1) * 3.0;
        2.0 * entry + accumulation_growth(8) * distortion
    }

    /// The three-projector sum's distance from the identity, as exact evidence over the one
    /// artifact evaluated, with its derived rounding bound.
    fn projector_evidence(
        outputs: &[f64; 4],
        native: &[f64; 4],
    ) -> Result<EvidenceStatus<(), &'static str>, String> {
        let value = frobenius_distance(outputs, native);
        EvidenceStatus::exact(
            value,
            projector_distance_roundoff(3, value),
            ExactBasis::Exhaustive { cardinality: 1 },
            None,
            "P11 projector sum of one decoded artifact",
        )
        .map_err(|error| error.to_string())
    }

    fn squared_distance(outputs: &[f64], native: &[f64]) -> f64 {
        outputs
            .iter()
            .zip(native)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
    }

    fn lattice_evidence(
        value: f64,
        numerical_error: f64,
    ) -> Result<EvidenceStatus<usize, &'static str>, String> {
        EvidenceStatus::exact(
            value,
            numerical_error,
            ExactBasis::Exhaustive { cardinality: 2 },
            None,
            "two lattice coordinates",
        )
        .map_err(|error| error.to_string())
    }

    #[test]
    fn fidelity_is_measured_on_the_decoded_artifact_not_on_the_parameters_it_came_from() {
        // P11: (2/C) v(t_c) v(t_c)^T at t_c = c·pi/C sums to I_2 on RP1.
        let angles: Vec<f64> = (0..3).map(|component| f64::from(component) * PI / 3.0).collect();
        let identity = [1.0, 0.0, 0.0, 1.0];
        let tolerance = 0.1;
        let quotient = PeriodicQuotient::new(PI).expect("the RP1 period is admissible");
        let evaluate = |decoded: &Vec<f64>| Ok::<[f64; 4], String>(projector_sum(decoded));

        // Positive control: the parameters before coding reproduce the identity, so an
        // evaluation that read them would be certified.
        let undecoded =
            projector_evidence(&projector_sum(&angles), &identity).expect("undecoded evidence");
        assert!(undecoded.certifies_at_most(tolerance), "{undecoded:?}");

        // At 2 bits the decoded angles are 0, pi/4 and 3pi/4, whose projector sum
        // misses the identity by sqrt(2)/3.
        let coarse = QuotientCode::encode(&angles, quotient, 2).expect("angles encode");
        assert_eq!(coarse.indices(), &[0_u64, 1, 3]);
        let coarse_fidelity =
            decode_then_evaluate(&coarse, evaluate, &identity, projector_evidence, tolerance)
                .expect("the coarse artifact evaluates");
        assert_eq!(
            coarse_fidelity.verdict(),
            FidelityVerdict::Violates,
            "{:?}",
            coarse_fidelity.status()
        );

        let fine = QuotientCode::encode(&angles, quotient, 20).expect("angles encode");
        let fine_fidelity =
            decode_then_evaluate(&fine, evaluate, &identity, projector_evidence, tolerance)
                .expect("the fine artifact evaluates");
        assert_eq!(
            fine_fidelity.verdict(),
            FidelityVerdict::Meets,
            "{:?}",
            fine_fidelity.status()
        );
    }

    #[test]
    fn decode_then_evaluate_refuses_an_undeclared_tolerance_or_a_refused_distortion() {
        let precision = DeclaredPrecision::new(2).expect("precision in range");
        let code = LatticeCode::encode(&[0.25, -1.5], precision).expect("values encode");
        let reference = vec![0.25, -1.5];
        let evaluate = |decoded: &Vec<f64>| Ok::<Vec<f64>, String>(decoded.clone());
        // Both values lie on the lattice and decode exactly, so the squared error is exact.
        let squared_error = |outputs: &Vec<f64>, native: &Vec<f64>| {
            lattice_evidence(squared_distance(outputs, native), 0.0)
        };

        // Positive control: an exact zero figure meets a zero tolerance.
        let exact = decode_then_evaluate(&code, evaluate, &reference, squared_error, 0.0)
            .expect("a declared tolerance evaluates");
        assert_eq!(exact.status().upper_bound(), Some(0.0));
        assert_eq!(exact.verdict(), FidelityVerdict::Meets);

        for tolerance in [f64::NAN, -1.0, f64::INFINITY] {
            assert!(
                decode_then_evaluate(&code, evaluate, &reference, squared_error, tolerance).is_err(),
                "tolerance {tolerance} was admitted"
            );
        }
        // The status constructor refuses a NaN figure and a negative rounding bound, and the
        // refusal propagates instead of being compared.
        let not_a_number = |outputs: &Vec<f64>, native: &Vec<f64>| {
            lattice_evidence(squared_distance(outputs, native) + f64::NAN, 0.0)
        };
        assert!(decode_then_evaluate(&code, evaluate, &reference, not_a_number, 0.0).is_err());
        let negative_error = |outputs: &Vec<f64>, native: &Vec<f64>| {
            lattice_evidence(squared_distance(outputs, native), -1.0)
        };
        assert!(decode_then_evaluate(&code, evaluate, &reference, negative_error, 0.0).is_err());
        let refused = |decoded: &Vec<f64>| {
            Err::<Vec<f64>, String>(format!("native execution refused {} values", decoded.len()))
        };
        assert!(decode_then_evaluate(&code, refused, &reference, squared_error, 0.0).is_err());
    }

    #[test]
    fn a_verdict_is_only_what_the_status_proves_about_the_tolerance() {
        let fidelity =
            |status: EvidenceStatus<usize, &'static str>| DecodedFidelity { status, tolerance: 0.25 };
        let exact = |value: f64, numerical_error: f64| {
            lattice_evidence(value, numerical_error).expect("a finite figure and error")
        };
        // An exact figure at the tolerance meets it. The same figure with a rounding band
        // neither meets nor violates it.
        assert_eq!(fidelity(exact(0.25, 0.0)).verdict(), FidelityVerdict::Meets);
        assert_eq!(
            fidelity(exact(0.25, UNIT_ROUNDOFF * 0.25)).verdict(),
            FidelityVerdict::Unresolved
        );
        assert_eq!(fidelity(exact(0.3, 0.01)).verdict(), FidelityVerdict::Violates);
        assert_eq!(fidelity(exact(0.2, 0.01)).verdict(), FidelityVerdict::Meets);
        // A derived uniform bound certifies from above.
        let bound = EvidenceStatus::uniform_bound(0.2, 0.0, "two lattice coordinates")
            .expect("a finite bound");
        assert_eq!(fidelity(bound).verdict(), FidelityVerdict::Meets);
        // Negative control: a statistical estimate proves no bound, so even a zero estimate
        // is unresolved.
        let estimate = EvidenceStatus::statistical_estimate(0.0, 0.0, 10, "two lattice coordinates")
            .expect("an estimate");
        assert_eq!(fidelity(estimate).verdict(), FidelityVerdict::Unresolved);
    }

    /// `L_int(value)` for a positive integer: the prefix code length a message pays.
    fn prefix_bits(value: u64) -> u64 {
        crate::codec::prefix_integer_len_bits(value)
            .expect("a positive integer has a prefix codeword")
    }

    /// `L_s(value)`: the signed prefix code length of a lattice index or precision exponent.
    fn signed_bits(value: i64) -> u64 {
        signed_prefix_integer_len_bits(value)
            .expect("a representable integer has a signed codeword")
    }

    #[test]
    fn lattice_message_reads_back_at_its_exact_prefix_code_length() {
        let cases: [(i32, &[i64]); 4] = [
            (3, &[0, -1, 1, 1 << 20, -(1 << 53)]),
            (0, &[5, -5]),
            (EXPONENT_LIMIT, &[0, -1, 1, 3]),
            (-EXPONENT_LIMIT, &[0, -1, 1, 3]),
        ];
        for (fraction_bits, indices) in cases {
            let precision = DeclaredPrecision::new(fraction_bits).expect("precision in range");
            let code = LatticeCode::from_indices(precision, indices.to_vec())
                .expect("indices decode exactly");
            let mut message = BitString::new();
            code.write(&mut message).expect("a lattice code writes");
            let expected_bits = prefix_bits(indices.len() as u64 + 1)
                + signed_bits(i64::from(fraction_bits))
                + indices.iter().map(|&index| signed_bits(index)).sum::<u64>();
            assert_eq!(message.len_bits(), expected_bits);
            // The header is the count and the declared precision. index_bits is the rest.
            let header =
                prefix_bits(indices.len() as u64 + 1) + signed_bits(i64::from(fraction_bits));
            assert_eq!(message.len_bits(), header + code.index_bits().expect("index bits"));
            let mut reader = message.reader();
            let decoded = LatticeCode::read(&mut reader).expect("the message reads back");
            assert!(reader.finish().is_ok());
            assert_eq!(decoded, code);
        }
    }

    #[test]
    fn lattice_index_bits_count_the_index_codewords_and_not_the_header() {
        let unit = DeclaredPrecision::new(0).expect("unit step");
        // Hand pin: the omega lengths of zigzag(k) + 1 for k = 0, -1, 1, -2 are 1, 3, 3, 6.
        let code = LatticeCode::from_indices(unit, vec![0, -1, 1, -2]).expect("indices decode");
        assert_eq!(code.index_bits(), Ok(13));
        // An empty code has no index bits, but its message still carries its header: count 0
        // as omega(1), 1 bit, and precision 0 as omega(zigzag(0) + 1), 1 bit.
        let empty = LatticeCode::from_indices(unit, Vec::new()).expect("an empty code");
        assert_eq!(empty.index_bits(), Ok(0));
        let mut message = BitString::new();
        empty.write(&mut message).expect("an empty code writes");
        assert_eq!(message.len_bits(), 2);
    }

    #[test]
    fn a_lattice_message_cut_short_or_announcing_more_indices_than_bits_is_refused() {
        let precision = DeclaredPrecision::new(2).expect("precision in range");
        let code = LatticeCode::from_indices(precision, vec![3, -7, 11]).expect("indices decode");
        let mut message = BitString::new();
        code.write(&mut message).expect("a lattice code writes");
        // Positive control: the whole message reads back.
        assert!(LatticeCode::read(&mut message.reader()).is_ok());

        let mut short = BitString::new();
        let mut reader = message.reader();
        for _ in 1..message.len_bits() {
            short.push_bit(reader.read_bit().expect("inside the message"));
        }
        assert!(LatticeCode::read(&mut short.reader()).is_err());

        // A header announcing 1000 indices followed by 8 bits is refused before allocating.
        let mut hostile = BitString::new();
        encode_prefix_integer(&mut hostile, 1000 + 1).expect("count codeword");
        encode_signed_prefix_integer(&mut hostile, 2).expect("precision codeword");
        hostile.push_bits(0xFF, 8).expect("eight bits");
        let refusal = LatticeCode::read(&mut hostile.reader()).expect_err("count beyond the message");
        assert!(refusal.contains("remain"), "{refusal}");
    }

    #[test]
    fn quotient_message_is_its_header_plus_exactly_its_index_bits() {
        let quotient = PeriodicQuotient::new(PI).expect("the RP1 period is admissible");
        let labels = [0.0, 1.0, 2.5, -1.0];
        for resolution_bits in [1, 10, 50] {
            let code = QuotientCode::encode(&labels, quotient, resolution_bits).expect("labels encode");
            let mut message = BitString::new();
            code.write(&mut message).expect("a quotient code writes");
            let header =
                prefix_bits(labels.len() as u64 + 1) + prefix_bits(u64::from(resolution_bits) + 1);
            assert_eq!(message.len_bits(), header + code.index_bits());
            let mut reader = message.reader();
            let decoded = QuotientCode::read(&mut reader, quotient).expect("the message reads back");
            assert!(reader.finish().is_ok());
            assert_eq!(decoded, code);
        }

        // Count guard at b = 10: 30 payload bits hold three indices, 29 do not.
        for (payload_bits, admitted) in [(30, true), (29, false)] {
            let mut message = BitString::new();
            encode_prefix_integer(&mut message, 3 + 1).expect("count codeword");
            encode_prefix_integer(&mut message, 10 + 1).expect("resolution codeword");
            message.push_bits(0, payload_bits).expect("payload bits");
            assert_eq!(QuotientCode::read(&mut message.reader(), quotient).is_ok(), admitted);
        }

        // A 0-bit resolution header is refused, while the smallest resolved one reads.
        for (resolution_bits, admitted) in [(0_u64, false), (1, true)] {
            let mut message = BitString::new();
            encode_prefix_integer(&mut message, 1 + 1).expect("count codeword");
            encode_prefix_integer(&mut message, resolution_bits + 1).expect("resolution codeword");
            message.push_bits(0, 1).expect("one payload bit");
            assert_eq!(QuotientCode::read(&mut message.reader(), quotient).is_ok(), admitted);
        }
    }
}
