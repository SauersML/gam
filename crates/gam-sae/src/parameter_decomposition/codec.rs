//! Code lengths of executable mechanism programs (#2951 P11, P18).
//!
//! A decomposition is scored by the length of a message that decodes to it, and
//! its fidelity is measured on what the decoder returns. Every length here is the
//! bit count of a message the paired decoder reads back, so a reported length is
//! a property of an implemented code, not a rate formula. The #2946 overclaim
//! audit (fr-census, issue comment 5716123817) separates exact code lengths (its
//! row S17) from Gaussian-surrogate and small-cell figures (row S16). The lengths
//! returned here are of the first kind. They are `u64` bit counts, never
//! [`crate::description_length::ScoredBits`], so they cannot be subtracted from a
//! surrogate figure.
//!
//! # Per-input versus global cost (P18)
//!
//! `E_x |S_x|` is not `|∪_x S_x|`: a decomposition whose inputs each use few
//! components can still need many components in total. The artifact is a global
//! library, paid once, plus one local packet per input
//! ([`LibraryPacketArtifact`]). A packet decodes from the library size and its own
//! bits alone ([`decode_support_packet`]), so each input is independently
//! executable. The library stores only the union of the supports
//! ([`union_support_library`]). There are no free real-valued coefficients: a real
//! enters a library only through a declared-precision code (`precision.rs`), and
//! distortion is measured on the decoded artifact.
//!
//! # Codes
//!
//! * An integer `n ≥ 1` is sent in the Elias omega prefix code
//!   ([`encode_prefix_integer`]). Its Kraft sum is 1.
//! * An index into an alphabet the decoder already knows, of size `M`, takes
//!   `⌈log₂ M⌉` bits ([`encode_fixed_index`]).
//! * A `k`-subset of `{0, …, n−1}` is sent as `k + 1` in the prefix code, then its
//!   colex rank in `⌈log₂ C(n, k)⌉` bits ([`encode_subset`]):
//!   `L(S) = L_int(k + 1) + ⌈log₂ C(n, k)⌉`. The rank is exact integer arithmetic,
//!   so it does not overflow at library widths.
//! * A program graph is an ordered DAG in topological order ([`encode_ordered_dag`]).
//!
//! # Identity versus a projector family (P11, A8)
//!
//! [`code_saving_at_proven_fidelity`] compares two decoded artifacts that both
//! meet one declared fidelity tolerance. Take `P_c = (2/C) v(t_c) v(t_c)ᵀ` at equally
//! spaced `t_c ∈ [0, π)` with `C ≥ 2`, so `Σ_c P_c = I₂` (at `C = 1` the family sums to
//! `diag(2, 0)`). For a unit `x`, `‖x − P_S x‖ ≥ xᵀ(x − P_S x) ≥ 1 − 2|S|/C`, so
//! reconstruction error `‖x − P_S x‖ ≤ ε` needs `|S| ≥ C(1 − ε)/2` on every input, and
//! a squared-error tolerance `‖x − P_S x‖² ≤ ε` needs `|S| ≥ C(1 − √ε)/2`. Each packet
//! then costs at least the least
//! subset code over those cardinalities, while the identity program needs no
//! packet at all. A smooth parameter family is not a good decomposition by itself.
//!
//! # Tensor addresses (P18)
//!
//! A parameter edit is addressed against one teacher's tensor registry (`lift`). The
//! address header declares the registry's fingerprint and the experiment unit's sequence
//! length ([`encode_address_header`]): a use site is the `k`-th read of a tensor on one
//! forward path, so an address means nothing against another registry, and the length is
//! the universe of declared positions. An edit scope (`occurrence`) is sent as two fixed
//! indices ([`encode_edit_scope`]): the storage tensor among the editable ones, then the
//! global edit or one of that tensor's use sites, both in the registry's id order. That
//! index is not a use-site ordinal: ids sort as strings, so `w#10` precedes `w#2`. A
//! position scope takes one bit for every position, or that bit and the enumerative
//! subset code of the declared positions ([`encode_position_scope`]).

use std::cmp::Ordering;
use std::fmt;

use super::lift::{TeacherFingerprint, TensorId, TensorRegistry};
use super::occurrence::{EditScope, OccurrenceError, ParameterEditRecord, PositionScope};
use super::precision::{DecodedFidelity, FidelityVerdict};
use super::supports::CardinalityCode;

/// Why a message could not be written or read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// The reader needed more bits than the message holds.
    UnexpectedEnd { needed: u64, remaining: u64 },
    /// Bits remained after one complete codeword, so the message is not the
    /// codeword of one object.
    TrailingBits { remaining: u64 },
    /// The bits are not a codeword of the declared code.
    InvalidCodeword(String),
    /// The object violates the code's contract, so it has no codeword.
    InvalidInput(String),
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd { needed, remaining } => write!(
                formatter,
                "message ended: {needed} bits needed, {remaining} remain"
            ),
            Self::TrailingBits { remaining } => write!(
                formatter,
                "{remaining} bits left unread after one complete codeword"
            ),
            Self::InvalidCodeword(reason) => write!(formatter, "not a codeword: {reason}"),
            Self::InvalidInput(reason) => write!(formatter, "no codeword: {reason}"),
        }
    }
}

impl std::error::Error for CodecError {}

/// An append-only bit string, most significant bit of each byte first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitString {
    bytes: Vec<u8>,
    len_bits: u64,
}

impl BitString {
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            len_bits: 0,
        }
    }

    /// The exact length of the message in bits.
    pub fn len_bits(&self) -> u64 {
        self.len_bits
    }

    pub fn is_empty(&self) -> bool {
        self.len_bits == 0
    }

    pub fn push_bit(&mut self, bit: bool) {
        let offset = self.len_bits % 8;
        if offset == 0 {
            self.bytes.push(0);
        }
        if bit {
            if let Some(last) = self.bytes.last_mut() {
                *last |= 0x80 >> offset;
            }
        }
        self.len_bits += 1;
    }

    /// Append the low `width` bits of `value`, most significant first. A value
    /// with a set bit at or above `width` has no `width`-bit codeword.
    pub fn push_bits(&mut self, value: u64, width: u32) -> Result<(), CodecError> {
        if width > u64::BITS || (width < u64::BITS && value >> width != 0) {
            return Err(CodecError::InvalidInput(format!(
                "{value} does not fit in {width} bits"
            )));
        }
        for position in (0..width).rev() {
            self.push_bit((value >> position) & 1 == 1);
        }
        Ok(())
    }

    /// Append another message after this one.
    pub fn append(&mut self, other: &BitString) {
        for index in 0..other.len_bits {
            self.push_bit(other.bit(index));
        }
    }

    pub fn reader(&self) -> BitReader<'_> {
        BitReader {
            bits: self,
            position: 0,
        }
    }

    fn bit(&self, index: u64) -> bool {
        self.bytes[(index / 8) as usize] & (0x80 >> (index % 8)) != 0
    }
}

/// A cursor over a [`BitString`].
#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    bits: &'a BitString,
    position: u64,
}

impl BitReader<'_> {
    pub fn remaining_bits(&self) -> u64 {
        self.bits.len_bits - self.position
    }

    fn require(&self, needed: u64) -> Result<(), CodecError> {
        let remaining = self.remaining_bits();
        if needed > remaining {
            return Err(CodecError::UnexpectedEnd { needed, remaining });
        }
        Ok(())
    }

    pub fn read_bit(&mut self) -> Result<bool, CodecError> {
        self.require(1)?;
        let bit = self.bits.bit(self.position);
        self.position += 1;
        Ok(bit)
    }

    /// Read `width ≤ 64` bits, most significant first.
    pub fn read_bits(&mut self, width: u32) -> Result<u64, CodecError> {
        if width > u64::BITS {
            return Err(CodecError::InvalidInput(format!(
                "a {width}-bit field exceeds u64"
            )));
        }
        self.require(u64::from(width))?;
        let mut value = 0_u64;
        for _ in 0..width {
            value = (value << 1) | u64::from(self.bits.bit(self.position));
            self.position += 1;
        }
        Ok(value)
    }

    /// Refuse a message with bits left after its codeword.
    pub fn finish(self) -> Result<(), CodecError> {
        match self.remaining_bits() {
            0 => Ok(()),
            remaining => Err(CodecError::TrailingBits { remaining }),
        }
    }
}

/// A nonnegative integer in little-endian 64-bit limbs with no trailing zero limb.
/// It carries an enumerative subset rank, whose range `C(n, k)` exceeds every
/// machine integer at library widths.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Natural {
    limbs: Vec<u64>,
}

impl Natural {
    fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    fn one() -> Self {
        Self { limbs: vec![1] }
    }

    fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    fn mul_small(&mut self, factor: u64) {
        let mut carry = 0_u128;
        for limb in &mut self.limbs {
            let product = u128::from(*limb) * u128::from(factor) + carry;
            *limb = product as u64;
            carry = product >> u64::BITS;
        }
        if carry != 0 {
            self.limbs.push(carry as u64);
        }
        self.trim();
    }

    /// Divide by a nonzero `divisor` that divides `self`. Every call site divides a
    /// product the binomial recurrences prove to be a multiple of the divisor.
    fn div_exact(&mut self, divisor: u64) {
        let mut remainder = 0_u128;
        for limb in self.limbs.iter_mut().rev() {
            let current = (remainder << u64::BITS) | u128::from(*limb);
            *limb = (current / u128::from(divisor)) as u64;
            remainder = current % u128::from(divisor);
        }
        self.trim();
    }

    fn add_assign(&mut self, other: &Natural) {
        if self.limbs.len() < other.limbs.len() {
            self.limbs.resize(other.limbs.len(), 0);
        }
        let mut carry = false;
        for (index, limb) in self.limbs.iter_mut().enumerate() {
            let addend = other.limbs.get(index).copied().unwrap_or(0);
            if !carry && index >= other.limbs.len() {
                break;
            }
            let (sum, first) = limb.overflowing_add(addend);
            let (sum, second) = sum.overflowing_add(u64::from(carry));
            *limb = sum;
            carry = first || second;
        }
        if carry {
            self.limbs.push(1);
        }
    }

    /// `self −= other` for `other ≤ self`.
    fn sub_assign(&mut self, other: &Natural) {
        let mut borrow = false;
        for (index, limb) in self.limbs.iter_mut().enumerate() {
            let subtrahend = other.limbs.get(index).copied().unwrap_or(0);
            if !borrow && index >= other.limbs.len() {
                break;
            }
            let (difference, first) = limb.overflowing_sub(subtrahend);
            let (difference, second) = difference.overflowing_sub(u64::from(borrow));
            *limb = difference;
            borrow = first || second;
        }
        self.trim();
    }

    fn bit_len(&self) -> u64 {
        match self.limbs.last() {
            None => 0,
            Some(top) => {
                u64::from(u64::BITS) * (self.limbs.len() as u64 - 1)
                    + u64::from(u64::BITS - top.leading_zeros())
            }
        }
    }

    fn bit(&self, index: u64) -> bool {
        self.limbs
            .get((index / u64::from(u64::BITS)) as usize)
            .is_some_and(|limb| (limb >> (index % u64::from(u64::BITS))) & 1 == 1)
    }

    /// `⌈log₂ self⌉` for `self ≥ 1`: the width of a fixed index into `self` values.
    fn index_width(&self) -> u64 {
        let mut largest = self.clone();
        largest.sub_assign(&Natural::one());
        largest.bit_len()
    }

    fn write(&self, out: &mut BitString, width: u64) {
        for position in (0..width).rev() {
            out.push_bit(self.bit(position));
        }
    }

    fn read(reader: &mut BitReader<'_>, width: u64) -> Result<Natural, CodecError> {
        reader.require(width)?;
        let limb_bits = u64::from(u64::BITS);
        let mut limbs = vec![0_u64; width.div_ceil(limb_bits) as usize];
        for position in (0..width).rev() {
            if reader.read_bit()? {
                limbs[(position / limb_bits) as usize] |= 1 << (position % limb_bits);
            }
        }
        let mut value = Natural { limbs };
        value.trim();
        Ok(value)
    }
}

impl PartialOrd for Natural {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Natural {
    fn cmp(&self, other: &Self) -> Ordering {
        self.limbs
            .len()
            .cmp(&other.limbs.len())
            .then_with(|| self.limbs.iter().rev().cmp(other.limbs.iter().rev()))
    }
}

/// `C(n, k)` exactly. After step `j` the value is `C(n, j + 1)`, and
/// `C(n, j)·(n − j) = C(n, j + 1)·(j + 1)`, so every division is exact.
fn binomial(n: u64, k: u64) -> Natural {
    if k > n {
        return Natural::zero();
    }
    let mut value = Natural::one();
    for step in 0..k.min(n - k) {
        value.mul_small(n - step);
        value.div_exact(step + 1);
    }
    value
}

/// The colex rank `Σ_i C(c_i, i)` (1-based `i`) of an ascending subset.
///
/// The running coefficient moves along two exact recurrences,
/// `C(c + 1, i) = C(c, i)·(c + 1)/(c + 1 − i)` and
/// `C(c, i + 1) = C(c, i)·(c − i)/(i + 1)`, which stay nonzero once `c_i ≥ i`.
/// Before that the subset is the prefix `{0, …, j − 1}`, whose terms are zero.
fn colex_rank(elements: &[usize]) -> Natural {
    let mut rank = Natural::zero();
    let Some(first) = elements
        .iter()
        .enumerate()
        .position(|(position, &element)| element != position)
    else {
        return rank;
    };
    let mut index = first as u64 + 1;
    let mut element = elements[first] as u64;
    let mut coefficient = binomial(element, index);
    rank.add_assign(&coefficient);
    for &next in &elements[first + 1..] {
        let next = next as u64;
        while element < next {
            coefficient.mul_small(element + 1);
            coefficient.div_exact(element + 1 - index);
            element += 1;
        }
        coefficient.mul_small(element - index);
        coefficient.div_exact(index + 1);
        index += 1;
        rank.add_assign(&coefficient);
    }
    rank
}

/// The ascending `cardinality`-subset of `{0, …, universe − 1}` with colex rank
/// `rank < C(universe, cardinality)`.
///
/// Greedy from the largest element: `c_i` is the largest `c` with `C(c, i) ≤ r`.
/// The coefficient walks down along `C(c − 1, i) = C(c, i)·(c − i)/c`, and moves to
/// the next index along `C(c − 1, i − 1) = C(c, i)·i/c`.
fn colex_unrank(universe: u64, cardinality: u64, mut rank: Natural) -> Vec<usize> {
    let mut elements = vec![0_usize; cardinality as usize];
    if cardinality == 0 {
        return elements;
    }
    let mut index = cardinality;
    let mut element = universe - 1;
    let mut coefficient = binomial(element, index);
    loop {
        // A coefficient above the rank is positive, so `element ≥ index ≥ 1`.
        while coefficient > rank {
            coefficient.mul_small(element - index);
            coefficient.div_exact(element);
            element -= 1;
        }
        elements[(index - 1) as usize] = element as usize;
        rank.sub_assign(&coefficient);
        if index == 1 {
            break;
        }
        if coefficient.is_zero() {
            // `C(c, i) = 0` means `c = i − 1`, which forces `{0, …, i − 2}` below it.
            for position in 0..(index - 1) as usize {
                elements[position] = position;
            }
            break;
        }
        coefficient.mul_small(index);
        coefficient.div_exact(element);
        element -= 1;
        index -= 1;
    }
    elements
}

/// Length in bits of the Elias omega codeword of `value ≥ 1`: one terminating bit
/// plus the width of every group `n → ⌊log₂ n⌋` until the group is 1.
pub fn prefix_integer_len_bits(value: u64) -> Result<u64, CodecError> {
    if value == 0 {
        return Err(CodecError::InvalidInput(
            "the prefix integer code covers integers from 1".to_string(),
        ));
    }
    let mut bits = 1_u64;
    let mut group = value;
    while group > 1 {
        let width = u64::from(u64::BITS - group.leading_zeros());
        bits += width;
        group = width - 1;
    }
    Ok(bits)
}

/// Write `value ≥ 1` in the Elias omega code.
///
/// The codeword is the binary groups `n, ⌊log₂ n⌋, ⌊log₂⌊log₂ n⌋⌋, …` down to (not
/// including) 1, written in reverse order, then a 0. Each group begins with a 1, so
/// the decoder reads a 1 as "a group of the current value plus one bits follows"
/// and a 0 as the end. A random bit stream ends with probability 1, so the Kraft
/// sum is 1.
pub fn encode_prefix_integer(out: &mut BitString, value: u64) -> Result<(), CodecError> {
    if value == 0 {
        return Err(CodecError::InvalidInput(
            "the prefix integer code covers integers from 1".to_string(),
        ));
    }
    let mut groups = Vec::new();
    let mut group = value;
    while group > 1 {
        let width = u64::BITS - group.leading_zeros();
        groups.push((group, width));
        group = u64::from(width - 1);
    }
    for &(group, width) in groups.iter().rev() {
        out.push_bits(group, width)?;
    }
    out.push_bit(false);
    Ok(())
}

/// Read one Elias omega codeword ([`encode_prefix_integer`]).
pub fn decode_prefix_integer(reader: &mut BitReader<'_>) -> Result<u64, CodecError> {
    let mut value = 1_u64;
    while reader.read_bit()? {
        // The 1 just read leads a group of `value + 1` bits, which holds the next value.
        if value >= u64::from(u64::BITS) {
            return Err(CodecError::InvalidCodeword(format!(
                "a {}-bit group exceeds u64",
                value + 1
            )));
        }
        let low = reader.read_bits(value as u32)?;
        value = (1_u64 << value) | low;
    }
    Ok(value)
}

/// The zigzag index `2k` for `k ≥ 0` and `2|k| − 1` for `k < 0`, a bijection of `i64` onto `u64`.
fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> (i64::BITS - 1))) as u64
}

/// The inverse of [`zigzag`]. `index >> 1` is below `2^63`, so the cast is exact.
fn unzigzag(index: u64) -> i64 {
    let half = (index >> 1) as i64;
    if index & 1 == 0 { half } else { -half - 1 }
}

/// `zigzag(value) + 1`, the argument of the unsigned prefix code. `i64::MIN` has zigzag index
/// `u64::MAX`, which leaves no room for the `+ 1`, so it has no codeword.
fn signed_codeword_argument(value: i64) -> Result<u64, CodecError> {
    zigzag(value).checked_add(1).ok_or_else(|| {
        CodecError::InvalidInput("i64::MIN has no signed prefix codeword".to_string())
    })
}

/// Length in bits of the signed prefix codeword of `value`: [`prefix_integer_len_bits`] of
/// `zigzag(value) + 1`. `i64::MIN` is refused.
pub fn signed_prefix_integer_len_bits(value: i64) -> Result<u64, CodecError> {
    prefix_integer_len_bits(signed_codeword_argument(value)?)
}

/// Write `value` as the Elias omega codeword of `zigzag(value) + 1`. `i64::MIN` is refused
/// before any bit is written.
pub fn encode_signed_prefix_integer(out: &mut BitString, value: i64) -> Result<(), CodecError> {
    encode_prefix_integer(out, signed_codeword_argument(value)?)
}

/// Read one signed prefix codeword ([`encode_signed_prefix_integer`]). Every codeword decodes
/// into `[i64::MIN + 1, i64::MAX]`.
pub fn decode_signed_prefix_integer(reader: &mut BitReader<'_>) -> Result<i64, CodecError> {
    Ok(unzigzag(decode_prefix_integer(reader)? - 1))
}

/// `⌈log₂ M⌉`: the width of a fixed index into an alphabet of `M ≥ 1` symbols the
/// decoder already knows. A one-symbol alphabet costs zero bits.
pub fn fixed_index_len_bits(alphabet_size: usize) -> Result<u32, CodecError> {
    if alphabet_size == 0 {
        return Err(CodecError::InvalidInput(
            "an empty alphabet has no codewords".to_string(),
        ));
    }
    Ok(usize::BITS - (alphabet_size - 1).leading_zeros())
}

pub fn encode_fixed_index(
    out: &mut BitString,
    index: usize,
    alphabet_size: usize,
) -> Result<(), CodecError> {
    let width = fixed_index_len_bits(alphabet_size)?;
    if index >= alphabet_size {
        return Err(CodecError::InvalidInput(format!(
            "index {index} is outside an alphabet of {alphabet_size}"
        )));
    }
    out.push_bits(index as u64, width)
}

pub fn decode_fixed_index(
    reader: &mut BitReader<'_>,
    alphabet_size: usize,
) -> Result<usize, CodecError> {
    let width = fixed_index_len_bits(alphabet_size)?;
    let index = reader.read_bits(width)?;
    if index >= alphabet_size as u64 {
        return Err(CodecError::InvalidCodeword(format!(
            "index {index} is outside an alphabet of {alphabet_size}"
        )));
    }
    Ok(index as usize)
}

/// `L(S) = L_int(k + 1) + ⌈log₂ C(n, k)⌉` for a `k`-subset of `{0, …, n − 1}`.
///
/// The binomial is exact, at `O(min(k, n − k))` limb-vector multiplications.
pub fn subset_code_len_bits(universe: usize, cardinality: usize) -> Result<u64, CodecError> {
    if cardinality > universe {
        return Err(CodecError::InvalidInput(format!(
            "a {cardinality}-subset of {universe} elements does not exist"
        )));
    }
    let cardinality_bits = prefix_integer_len_bits(cardinality as u64 + 1)?;
    Ok(cardinality_bits + binomial(universe as u64, cardinality as u64).index_width())
}

/// Write an ascending subset of `{0, …, universe − 1}` in the enumerative code:
/// `k + 1` in the prefix integer code, then the colex rank in
/// `⌈log₂ C(universe, k)⌉` bits.
pub fn encode_subset(
    out: &mut BitString,
    universe: usize,
    elements: &[usize],
) -> Result<(), CodecError> {
    if let Some(pair) = elements.windows(2).find(|pair| pair[0] >= pair[1]) {
        return Err(CodecError::InvalidInput(format!(
            "subset elements must be strictly ascending, got {} then {}",
            pair[0], pair[1]
        )));
    }
    if let Some(&largest) = elements.last() {
        if largest >= universe {
            return Err(CodecError::InvalidInput(format!(
                "element {largest} is outside a universe of {universe}"
            )));
        }
    }
    let cardinality = elements.len() as u64;
    encode_prefix_integer(out, cardinality + 1)?;
    let width = binomial(universe as u64, cardinality).index_width();
    colex_rank(elements).write(out, width);
    Ok(())
}

/// Read one subset of `{0, …, universe − 1}` ([`encode_subset`]), ascending.
///
/// A cardinality whose rank field cannot fit in the remaining bits is refused before the
/// binomial is formed, so a short hostile message cannot force a large computation.
pub fn decode_subset(
    reader: &mut BitReader<'_>,
    universe: usize,
) -> Result<Vec<usize>, CodecError> {
    let cardinality = decode_prefix_integer(reader)? - 1;
    if cardinality > universe as u64 {
        return Err(CodecError::InvalidCodeword(format!(
            "a {cardinality}-subset of {universe} elements does not exist"
        )));
    }
    // With `j = min(k, n − k) ≥ 1`, `C(n, k) = Π_{i=1..j} (n − j + i)/i` and every factor is at
    // least 2 (because `n − j ≥ j ≥ i`), so the rank field holds at least `j` bits.
    let smaller = cardinality.min(universe as u64 - cardinality);
    if smaller > reader.remaining_bits() {
        return Err(CodecError::InvalidCodeword(format!(
            "a {cardinality}-subset of {universe} elements needs at least {smaller} rank bits, \
             {} remain",
            reader.remaining_bits()
        )));
    }
    let count = binomial(universe as u64, cardinality);
    let rank = Natural::read(reader, count.index_width())?;
    if rank >= count {
        return Err(CodecError::InvalidCodeword(format!(
            "subset rank is not below C({universe}, {cardinality})"
        )));
    }
    Ok(colex_unrank(universe as u64, cardinality, rank))
}

/// One node of a program graph: a label from an alphabet the decoder knows, and an
/// ordered argument list naming earlier nodes. Order matters (`Compose` does not
/// commute) and an argument may repeat.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DagNode {
    pub label: usize,
    pub arguments: Vec<usize>,
}

/// Write a program graph in topological order.
///
/// The codeword is the node count plus one in the prefix integer code, then per node
/// its label as a fixed index into `label_alphabet`, and for every node after the
/// first its arity plus one in the prefix integer code and each argument as a fixed
/// index into `max(i, 2)` symbols, where `i` is the number of nodes before it. The
/// first node has no earlier node, so its empty argument list costs nothing. The
/// second node can only read the first, yet each of its arguments still spends one
/// bit: every argument then costs at least one bit, which lets a decoder bound an
/// arity by the bits left in the message.
pub fn encode_ordered_dag(
    out: &mut BitString,
    label_alphabet: usize,
    nodes: &[DagNode],
) -> Result<(), CodecError> {
    encode_prefix_integer(out, nodes.len() as u64 + 1)?;
    for (position, node) in nodes.iter().enumerate() {
        encode_fixed_index(out, node.label, label_alphabet)?;
        if position == 0 {
            if !node.arguments.is_empty() {
                return Err(CodecError::InvalidInput(
                    "the first node has no earlier node to read".to_string(),
                ));
            }
            continue;
        }
        encode_prefix_integer(out, node.arguments.len() as u64 + 1)?;
        for &argument in &node.arguments {
            if argument >= position {
                return Err(CodecError::InvalidInput(format!(
                    "node {position} reads node {argument}, which is not before it"
                )));
            }
            encode_fixed_index(out, argument, position.max(2))?;
        }
    }
    Ok(())
}

/// Read one program graph ([`encode_ordered_dag`]).
///
/// Every node after the first spends at least its one-bit arity codeword and every
/// argument at least one bit, so a node count above the remaining bits plus one, or an
/// arity above the remaining bits, is refused before any allocation.
pub fn decode_ordered_dag(
    reader: &mut BitReader<'_>,
    label_alphabet: usize,
) -> Result<Vec<DagNode>, CodecError> {
    let node_count = decode_prefix_integer(reader)? - 1;
    if node_count > reader.remaining_bits().saturating_add(1) {
        return Err(CodecError::InvalidCodeword(format!(
            "{node_count} nodes cannot fit in {} remaining bits",
            reader.remaining_bits()
        )));
    }
    let mut nodes = Vec::with_capacity(node_count as usize);
    for position in 0..node_count as usize {
        let label = decode_fixed_index(reader, label_alphabet)?;
        let mut arguments = Vec::new();
        if position > 0 {
            let arity = decode_prefix_integer(reader)? - 1;
            if arity > reader.remaining_bits() {
                return Err(CodecError::InvalidCodeword(format!(
                    "node {position} announces {arity} arguments in {} remaining bits",
                    reader.remaining_bits()
                )));
            }
            for _ in 0..arity {
                let argument = decode_fixed_index(reader, position.max(2))?;
                if argument >= position {
                    return Err(CodecError::InvalidCodeword(format!(
                        "node {position} reads node {argument}, which is not before it"
                    )));
                }
                arguments.push(argument);
            }
        }
        nodes.push(DagNode { label, arguments });
    }
    Ok(nodes)
}

/// A global library, paid once, and one independently executable local packet per
/// input (P18).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LibraryPacketArtifact {
    pub library: BitString,
    pub packets: Vec<BitString>,
}

impl LibraryPacketArtifact {
    /// The exact artifact length: the library plus every packet.
    pub fn total_bits(&self) -> u64 {
        self.library.len_bits() + self.packets.iter().map(BitString::len_bits).sum::<u64>()
    }
}

/// The union library of per-input supports, with each support re-indexed into it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SupportLibrary {
    /// The component ids the library stores, ascending: `|∪_x S_x|` of them.
    pub components: Vec<usize>,
    /// Each input's support as ascending positions in [`Self::components`].
    pub supports: Vec<Vec<usize>>,
    /// `Σ_x |S_x|`, whose mean `E_x |S_x|` is not the library size.
    pub summed_support: u64,
}

/// Build the union library of ascending per-input supports of component ids.
pub fn union_support_library(supports: &[Vec<usize>]) -> Result<SupportLibrary, CodecError> {
    for (input, support) in supports.iter().enumerate() {
        if let Some(pair) = support.windows(2).find(|pair| pair[0] >= pair[1]) {
            return Err(CodecError::InvalidInput(format!(
                "support of input {input} must be strictly ascending, got {} then {}",
                pair[0], pair[1]
            )));
        }
    }
    let mut components: Vec<usize> = supports.iter().flatten().copied().collect();
    components.sort_unstable();
    components.dedup();
    let remapped = supports
        .iter()
        .map(|support| {
            support
                .iter()
                .map(|&component| components.partition_point(|&stored| stored < component))
                .collect()
        })
        .collect();
    Ok(SupportLibrary {
        components,
        supports: remapped,
        summed_support: supports.iter().map(|support| support.len() as u64).sum(),
    })
}

/// One subset packet per input, each over library positions `{0, …, library_size − 1}`.
pub fn encode_support_packets(
    library_size: usize,
    supports: &[Vec<usize>],
) -> Result<Vec<BitString>, CodecError> {
    supports
        .iter()
        .map(|support| {
            let mut packet = BitString::new();
            encode_subset(&mut packet, library_size, support)?;
            Ok(packet)
        })
        .collect()
}

/// Decode one packet from the library size and its own bits alone. A packet with
/// bits left after its subset codeword is refused, so no packet can carry state
/// for another.
pub fn decode_support_packet(
    library_size: usize,
    packet: &BitString,
) -> Result<Vec<usize>, CodecError> {
    let mut reader = packet.reader();
    let support = decode_subset(&mut reader, library_size)?;
    reader.finish()?;
    Ok(support)
}

/// `reference − candidate` in bits between two decoded artifacts, each given with its exact
/// code length ([`LibraryPacketArtifact::total_bits`]) and the fidelity evidence of its decoded
/// outputs (positive means the candidate is shorter).
///
/// The comparison is refused unless both verdicts are [`FidelityVerdict::Meets`] under one
/// declared tolerance. Lengths at different fidelities are not a model comparison, and a figure
/// within its rounding band of the tolerance is `Unresolved`: it proves neither side, so it is
/// refused rather than read as a pass or a violation. Each verdict is stated over the inputs its
/// evaluation executed and says nothing beyond them.
pub fn code_saving_at_proven_fidelity<W, D, V, E>(
    reference: (u64, &DecodedFidelity<W, D>),
    candidate: (u64, &DecodedFidelity<V, E>),
) -> Result<i128, String> {
    let (reference_bits, reference_fidelity) = reference;
    let (candidate_bits, candidate_fidelity) = candidate;
    if reference_fidelity.tolerance().to_bits() != candidate_fidelity.tolerance().to_bits() {
        return Err(format!(
            "code comparison refused: the reference is scored at tolerance {} and the candidate \
             at {}, not one declared tolerance",
            reference_fidelity.tolerance(),
            candidate_fidelity.tolerance()
        ));
    }
    for (role, verdict) in [
        ("reference", reference_fidelity.verdict()),
        ("candidate", candidate_fidelity.verdict()),
    ] {
        if verdict != FidelityVerdict::Meets {
            return Err(format!(
                "code comparison refused: the {role} artifact's decoded fidelity is {verdict:?} \
                 at tolerance {}, not Meets",
                reference_fidelity.tolerance()
            ));
        }
    }
    Ok(i128::from(reference_bits) - i128::from(candidate_bits))
}

/// The enumerative subset code `L(S) = L_int(k + 1) + ⌈log₂ C(n, k)⌉` as the support code the
/// P12 minimum-code support search minimizes ([`super::supports::minimum_code_support`]).
///
/// The length is not monotone in `k`: near `k = n` the rank field vanishes and only the
/// cardinality codeword remains, so the cheapest admissible size can be the full support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnumerativeSubsetCode;

impl CardinalityCode for EnumerativeSubsetCode {
    type Error = CodecError;

    fn support_bits(&self, components: usize, size: usize) -> Result<u64, CodecError> {
        subset_code_len_bits(components, size)
    }
}

/// Writes the declarations an address message is read against, once per experiment unit:
/// the fingerprint of the teacher whose registry the addresses index (64 bits), then the
/// unit's sequence length, the universe of declared positions, in the prefix integer code.
/// Both are experiment inputs with no default. A sequence of length zero has no positions,
/// and it is refused.
pub fn encode_address_header(
    out: &mut BitString,
    registry: &TensorRegistry,
    length: usize,
) -> Result<(), CodecError> {
    address_header_len_bits(length)?;
    out.push_bits(registry.teacher_fingerprint().0, u64::BITS)?;
    encode_prefix_integer(out, length as u64)
}

/// Reads an address header ([`encode_address_header`]) against `registry`, refusing another
/// teacher's fingerprint. A matching fingerprint identifies the teacher; it certifies nothing.
pub fn decode_address_header(
    reader: &mut BitReader<'_>,
    registry: &TensorRegistry,
) -> Result<(TeacherFingerprint, usize), CodecError> {
    let written = reader.read_bits(u64::BITS)?;
    let expected = registry.teacher_fingerprint();
    if written != expected.0 {
        return Err(CodecError::InvalidCodeword(format!(
            "the message names teacher {written:#018x}, and the registry is {:#018x}",
            expected.0
        )));
    }
    let length = decode_prefix_integer(reader)?;
    let length = usize::try_from(length).map_err(|_| {
        CodecError::InvalidCodeword(format!("sequence length {length} does not fit this host"))
    })?;
    Ok((expected, length))
}

/// `64 + L_int(length)`: the exact length of an address header.
pub fn address_header_len_bits(length: usize) -> Result<u64, CodecError> {
    if length == 0 {
        return Err(CodecError::InvalidInput(
            "a sequence of length 0 has no positions to address".to_string(),
        ));
    }
    Ok(u64::from(u64::BITS) + prefix_integer_len_bits(length as u64)?)
}

/// The storage tensors an edit can change, in id order: matrix-shaped, because an edit is
/// held as matrix factors, and read by at least one use, because no experiment edits a
/// tensor nothing reads.
fn editable_storage(registry: &TensorRegistry) -> Vec<&TensorId> {
    registry
        .storage_ids()
        .filter(|id| {
            matches!(registry.storage(id), Some(tensor) if tensor.shape.len() == 2)
                && !registry.use_sites_of(id).is_empty()
        })
        .collect()
}

/// An edit scope's two fixed-index symbols: `(storage index, storage alphabet, scope index,
/// scope alphabet)`. `occurrence` validates the scope, so an alias named as storage, a
/// non-matrix storage, an unused tensor and an unknown use site are refused.
fn edit_scope_symbols(
    registry: &TensorRegistry,
    scope: &EditScope,
) -> Result<(usize, usize, usize, usize), CodecError> {
    let refused =
        |error: OccurrenceError| CodecError::InvalidInput(format!("edit scope refused: {error}"));
    scope.affected_uses(registry).map_err(refused)?;
    let (storage, _) = scope.storage(registry).map_err(refused)?;
    let editable = editable_storage(registry);
    let storage_index = editable.iter().position(|id| **id == storage).ok_or_else(|| {
        CodecError::InvalidInput(format!("{} is not an editable storage tensor", storage.0))
    })?;
    let sites = registry.use_sites_of(&storage);
    let scope_index = match scope {
        EditScope::Global(..) => 0,
        EditScope::UseSite(site) => {
            1 + sites.iter().position(|candidate| *candidate == site).ok_or_else(|| {
                CodecError::InvalidInput(format!("use site {} does not read {}", site.0, storage.0))
            })?
        }
    };
    Ok((storage_index, editable.len(), scope_index, 1 + sites.len()))
}

/// Writes an edit scope as a tensor address in two fixed indices: the storage tensor among
/// the editable storage tensors in id order (`⌈log₂ R⌉` bits), then the global edit or one
/// of that storage's use sites in id order (`⌈log₂(1 + U)⌉` bits). A global edit and a
/// use-specific edit of one tensor are different codewords, because they are different
/// experiments. The scope index is a position in the registry's id order, never a use-site
/// ordinal: ids sort as strings, so `w#10` precedes `w#2`.
pub fn encode_edit_scope(
    out: &mut BitString,
    registry: &TensorRegistry,
    scope: &EditScope,
) -> Result<(), CodecError> {
    let (storage_index, storage_count, scope_index, scope_count) = edit_scope_symbols(registry, scope)?;
    encode_fixed_index(out, storage_index, storage_count)?;
    encode_fixed_index(out, scope_index, scope_count)
}

/// Reads an edit scope ([`encode_edit_scope`]) against the registry the header named. An
/// index outside either alphabet is refused before any lookup.
pub fn decode_edit_scope(
    reader: &mut BitReader<'_>,
    registry: &TensorRegistry,
) -> Result<EditScope, CodecError> {
    let editable = editable_storage(registry);
    let storage = editable[decode_fixed_index(reader, editable.len())?];
    let sites = registry.use_sites_of(storage);
    Ok(match decode_fixed_index(reader, 1 + sites.len())? {
        0 => EditScope::Global(storage.clone()),
        index => EditScope::UseSite(sites[index - 1].clone()),
    })
}

/// `⌈log₂ R⌉ + ⌈log₂(1 + U)⌉`: the exact length of an edit scope.
pub fn edit_scope_len_bits(registry: &TensorRegistry, scope: &EditScope) -> Result<u64, CodecError> {
    let (_, storage_count, _, scope_count) = edit_scope_symbols(registry, scope)?;
    Ok(u64::from(fixed_index_len_bits(storage_count)?) + u64::from(fixed_index_len_bits(scope_count)?))
}

/// Writes a position scope over the header's sequence length: one bit for every position or
/// a declared set, then a declared set as the enumerative subset code over `0..length`. A
/// declared position at or beyond `length` is refused.
pub fn encode_position_scope(
    out: &mut BitString,
    scope: &PositionScope,
    length: usize,
) -> Result<(), CodecError> {
    scope
        .check_within(length)
        .map_err(|error| CodecError::InvalidInput(format!("position scope refused: {error}")))?;
    match scope.positions() {
        None => out.push_bit(false),
        Some(positions) => {
            out.push_bit(true);
            encode_subset(out, length, positions)?;
        }
    }
    Ok(())
}

/// Reads a position scope ([`encode_position_scope`]). A declared set is rebuilt through
/// `PositionScope::declared`, which refuses the empty set, and checked within `length`.
pub fn decode_position_scope(
    reader: &mut BitReader<'_>,
    length: usize,
) -> Result<PositionScope, CodecError> {
    if !reader.read_bit()? {
        return Ok(PositionScope::every());
    }
    let refused = |error: OccurrenceError| {
        CodecError::InvalidCodeword(format!("position scope refused: {error}"))
    };
    let scope = PositionScope::declared(decode_subset(reader, length)?).map_err(refused)?;
    scope.check_within(length).map_err(refused)?;
    Ok(scope)
}

/// `1`, or `1 + L_int(k + 1) + ⌈log₂ C(length, k)⌉` for `k` declared positions: the exact
/// length of a position scope.
pub fn position_scope_len_bits(scope: &PositionScope, length: usize) -> Result<u64, CodecError> {
    scope
        .check_within(length)
        .map_err(|error| CodecError::InvalidInput(format!("position scope refused: {error}")))?;
    match scope.positions() {
        None => Ok(1),
        Some(positions) => Ok(1 + subset_code_len_bits(length, positions.len())?),
    }
}

/// Writes where a parameter edit acts: its scope, then its positions. The edit's factors are
/// reals and go through `precision`. A record checked against a registry with another
/// fingerprint is refused, because its use-site ordinals number another forward path.
pub fn encode_edit_address(
    out: &mut BitString,
    registry: &TensorRegistry,
    length: usize,
    record: &ParameterEditRecord,
) -> Result<(), CodecError> {
    let expected = registry.teacher_fingerprint();
    if record.registry() != expected {
        return Err(CodecError::InvalidInput(format!(
            "the record was checked against teacher {:#018x}, and the registry is {:#018x}",
            record.registry().0,
            expected.0
        )));
    }
    encode_edit_scope(out, registry, record.scope())?;
    encode_position_scope(out, record.positions(), length)
}

/// Reads an edit address ([`encode_edit_address`]): the scope and positions a record is
/// rebuilt from, together with its decoded factors.
pub fn decode_edit_address(
    reader: &mut BitReader<'_>,
    registry: &TensorRegistry,
    length: usize,
) -> Result<(EditScope, PositionScope), CodecError> {
    let scope = decode_edit_scope(reader, registry)?;
    let positions = decode_position_scope(reader, length)?;
    Ok((scope, positions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::description_length::selection_bits;
    use crate::parameter_decomposition::apply::FactoredEdit;
    use crate::parameter_decomposition::lift::{TieOrientation, UseMap, UseSiteId};
    use crate::parameter_decomposition::precision::{
        DecodableArtifact, DecodedFidelity, FidelityVerdict, PeriodicQuotient, QuotientCode,
        decode_then_evaluate,
    };
    use crate::parameter_decomposition::supports::{
        CardinalityCode, ComponentSet, EvidenceStatus, EvidenceStatusError, ExactBasis,
        FailureHypergraph, SeparationOracle, minimum_code_support,
    };
    use gam_linalg::roundoff::accumulation_growth;
    use ndarray::{Array2, ArrayD};

    fn natural_to_u128(value: &Natural) -> u128 {
        assert!(value.limbs.len() <= 2, "value exceeds u128: {value:?}");
        value
            .limbs
            .iter()
            .enumerate()
            .map(|(index, &limb)| u128::from(limb) << (64 * index))
            .sum()
    }

    fn natural_from_u128(value: u128) -> Natural {
        let mut natural = Natural {
            limbs: vec![value as u64, (value >> 64) as u64],
        };
        natural.trim();
        natural
    }

    #[test]
    fn natural_arithmetic_matches_u128() {
        let samples = [
            0_u128,
            1,
            u128::from(u64::MAX),
            u128::from(u64::MAX) + 1,
            (1_u128 << 100) + 12_345,
            u128::MAX / 3,
        ];
        for &left in &samples {
            for &right in &samples {
                let (a, b) = (natural_from_u128(left), natural_from_u128(right));
                assert_eq!(a.cmp(&b), left.cmp(&right), "cmp {left} {right}");
                if let Some(sum) = left.checked_add(right) {
                    let mut total = a.clone();
                    total.add_assign(&b);
                    assert_eq!(natural_to_u128(&total), sum, "add {left} {right}");
                }
                if left >= right {
                    let mut difference = a.clone();
                    difference.sub_assign(&b);
                    assert_eq!(natural_to_u128(&difference), left - right, "sub {left} {right}");
                }
            }
            for factor in [0_u64, 1, 7, u64::from(u32::MAX)] {
                if let Some(product) = left.checked_mul(u128::from(factor)) {
                    let mut scaled = natural_from_u128(left);
                    scaled.mul_small(factor);
                    assert_eq!(natural_to_u128(&scaled), product, "mul {left} {factor}");
                    if factor > 0 {
                        scaled.div_exact(factor);
                        assert_eq!(natural_to_u128(&scaled), left, "div {product} {factor}");
                    }
                }
            }
            let natural = natural_from_u128(left);
            assert_eq!(natural.bit_len(), u64::from(128 - left.leading_zeros()));
        }
    }

    #[test]
    fn binomial_matches_exact_integer_recurrence() {
        // C(n, k) for n ≤ 120 stays below 2^128 through the recurrence's products.
        for n in 0_u64..=120 {
            let mut exact = 1_u128;
            for k in 0..=n {
                assert_eq!(natural_to_u128(&binomial(n, k)), exact, "C({n}, {k})");
                exact = exact * u128::from(n - k) / u128::from(k + 1);
            }
            assert!(binomial(n, n + 1).is_zero(), "C({n}, {}) must be zero", n + 1);
        }
    }

    #[test]
    fn prefix_integer_code_round_trips_and_is_prefix_free() {
        // Hand-derived codeword lengths: 1 → "0"; 2 → "10"+"0"; 4 → "10"+"100"+"0";
        // 16 → "10"+"100"+"10000"+"0".
        for (value, expected) in [(1_u64, 1_u64), (2, 3), (3, 3), (4, 6), (7, 6), (8, 7), (16, 11)] {
            assert_eq!(prefix_integer_len_bits(value), Ok(expected), "L_int({value})");
        }
        let mut values: Vec<u64> = (1..=4096).collect();
        values.extend([u64::from(u32::MAX) + 1, 1 << 63, u64::MAX]);
        let mut stream = BitString::new();
        for &value in &values {
            let before = stream.len_bits();
            encode_prefix_integer(&mut stream, value).expect("encode");
            assert_eq!(
                stream.len_bits() - before,
                prefix_integer_len_bits(value).expect("length"),
                "written length of {value}"
            );
        }
        // One concatenated stream decodes value by value only if no codeword is a
        // prefix of another.
        let mut reader = stream.reader();
        for &value in &values {
            assert_eq!(decode_prefix_integer(&mut reader), Ok(value));
        }
        assert_eq!(reader.finish(), Ok(()));
        // Kraft: Σ 2^(−L(n)) over n < 2^16 is at most 1, exactly in integers.
        let lengths: Vec<u64> = (1_u64..1 << 16)
            .map(|value| prefix_integer_len_bits(value).expect("length"))
            .collect();
        let longest = *lengths.iter().max().expect("nonempty");
        let kraft_numerator: u128 = lengths.iter().map(|&length| 1_u128 << (longest - length)).sum();
        assert!(
            kraft_numerator <= 1_u128 << longest,
            "Kraft sum exceeds 1: {kraft_numerator} / 2^{longest}"
        );
        assert!(matches!(
            prefix_integer_len_bits(0),
            Err(CodecError::InvalidInput(_))
        ));
    }

    #[test]
    fn prefix_integer_decoder_refuses_overflow_and_truncation() {
        // "1"+"1" gives 3, "1"+"111" gives 15, "1"+fifteen ones gives 65535, and the
        // next 1 announces a 65536-bit group.
        let mut overflow = BitString::new();
        overflow.push_bits(0b11, 2).expect("bits");
        overflow.push_bits(0b1111, 4).expect("bits");
        overflow.push_bits(0xFFFF, 16).expect("bits");
        overflow.push_bit(true);
        assert!(matches!(
            decode_prefix_integer(&mut overflow.reader()),
            Err(CodecError::InvalidCodeword(_))
        ));
        let mut truncated = BitString::new();
        encode_prefix_integer(&mut truncated, 1 << 40).expect("encode");
        let mut cut = BitString::new();
        for index in 0..truncated.len_bits() - 1 {
            cut.push_bit(truncated.bit(index));
        }
        assert!(matches!(
            decode_prefix_integer(&mut cut.reader()),
            Err(CodecError::UnexpectedEnd { .. })
        ));
    }

    #[test]
    fn signed_prefix_integer_code_round_trips_and_refuses_i64_min() {
        // Hand-derived: zigzag maps 0, −1, 1, −2 to 0, 1, 2, 3, so the codeword lengths are
        // L_int(1) = 1, L_int(2) = 3, L_int(3) = 3 and L_int(4) = 6.
        for (value, bits) in [(0_i64, 1_u64), (-1, 3), (1, 3), (-2, 6)] {
            assert_eq!(signed_prefix_integer_len_bits(value), Ok(bits), "L_s({value})");
        }
        let mut values: Vec<i64> = (-4096..=4096).collect();
        values.extend([i64::MAX, i64::MIN + 1, 1 << 53, -(1 << 53), 1022, -1022]);
        let mut stream = BitString::new();
        for &value in &values {
            let before = stream.len_bits();
            encode_signed_prefix_integer(&mut stream, value).expect("encode");
            assert_eq!(
                stream.len_bits() - before,
                signed_prefix_integer_len_bits(value).expect("length"),
                "written length of {value}"
            );
        }
        // One concatenated stream decodes value by value: the code stays prefix-free.
        let mut reader = stream.reader();
        for &value in &values {
            assert_eq!(decode_signed_prefix_integer(&mut reader), Ok(value));
        }
        assert_eq!(reader.finish(), Ok(()));
        // Positive control: i64::MIN has no codeword, and a refused encode writes nothing.
        assert!(matches!(
            signed_prefix_integer_len_bits(i64::MIN),
            Err(CodecError::InvalidInput(_))
        ));
        let mut refused = BitString::new();
        assert!(matches!(
            encode_signed_prefix_integer(&mut refused, i64::MIN),
            Err(CodecError::InvalidInput(_))
        ));
        assert!(refused.is_empty());
        // The two largest unsigned codewords decode to the ends of the signed range.
        let mut largest = BitString::new();
        encode_prefix_integer(&mut largest, u64::MAX).expect("encode");
        assert_eq!(decode_signed_prefix_integer(&mut largest.reader()), Ok(i64::MAX));
        let mut odd = BitString::new();
        encode_prefix_integer(&mut odd, u64::MAX - 1).expect("encode");
        assert_eq!(decode_signed_prefix_integer(&mut odd.reader()), Ok(i64::MIN + 1));
    }

    #[test]
    fn fixed_index_code_uses_ceiling_log_width() {
        for (alphabet, width) in [(1_usize, 0_u32), (2, 1), (3, 2), (4, 2), (5, 3), (1 << 20, 20)] {
            assert_eq!(fixed_index_len_bits(alphabet), Ok(width), "alphabet {alphabet}");
            let mut out = BitString::new();
            encode_fixed_index(&mut out, alphabet - 1, alphabet).expect("encode");
            assert_eq!(out.len_bits(), u64::from(width));
            assert_eq!(decode_fixed_index(&mut out.reader(), alphabet), Ok(alphabet - 1));
        }
        assert!(matches!(
            encode_fixed_index(&mut BitString::new(), 5, 5),
            Err(CodecError::InvalidInput(_))
        ));
        let mut unused = BitString::new();
        unused.push_bits(0b111, 3).expect("bits");
        assert!(matches!(
            decode_fixed_index(&mut unused.reader(), 5),
            Err(CodecError::InvalidCodeword(_))
        ));
    }

    #[test]
    fn subset_code_is_a_bijection_onto_colex_ranks() {
        for universe in 0_usize..=10 {
            let mut ranks_by_cardinality = vec![Vec::new(); universe + 1];
            for mask in 0_u32..(1 << universe) {
                let elements: Vec<usize> = (0..universe).filter(|&bit| mask >> bit & 1 == 1).collect();
                let cardinality = elements.len();
                let rank = natural_to_u128(&colex_rank(&elements));
                let count = natural_to_u128(&binomial(universe as u64, cardinality as u64));
                assert!(rank < count, "rank {rank} of {elements:?} not below C({universe}, {cardinality})");
                ranks_by_cardinality[cardinality].push(rank);
                let mut out = BitString::new();
                encode_subset(&mut out, universe, &elements).expect("encode");
                assert_eq!(
                    out.len_bits(),
                    subset_code_len_bits(universe, cardinality).expect("length"),
                    "written length of {elements:?}"
                );
                assert_eq!(decode_support_packet(universe, &out), Ok(elements.clone()));
            }
            for (cardinality, ranks) in ranks_by_cardinality.iter_mut().enumerate() {
                ranks.sort_unstable();
                let expected: Vec<u128> = (0..natural_to_u128(&binomial(universe as u64, cardinality as u64))).collect();
                assert_eq!(*ranks, expected, "ranks of {cardinality}-subsets of {universe}");
            }
        }
        // Colex endpoints: the lowest k elements rank 0, the highest k rank C(n, k) − 1.
        assert!(colex_rank(&[0, 1, 2]).is_zero());
        assert_eq!(natural_to_u128(&colex_rank(&[7, 8, 9])), 119);
        // The empty and full sets pay only their cardinality codeword.
        assert_eq!(subset_code_len_bits(16, 0), prefix_integer_len_bits(1));
        assert_eq!(subset_code_len_bits(16, 16), prefix_integer_len_bits(17));
    }

    #[test]
    fn subset_code_round_trips_at_library_width() {
        let universe = 100_000_usize;
        let cardinality = 300_usize;
        let stride = universe / cardinality;
        // Strictly ascending: consecutive elements differ by `stride` plus a residue
        // difference above `−stride`.
        let elements: Vec<usize> = (0..cardinality)
            .map(|position| position * stride + (position * position) % stride)
            .collect();
        let mut out = BitString::new();
        encode_subset(&mut out, universe, &elements).expect("encode");
        let length = subset_code_len_bits(universe, cardinality).expect("length");
        assert_eq!(out.len_bits(), length);
        assert_eq!(decode_support_packet(universe, &out), Ok(elements));
        // The rank field is ⌈log₂ C(n, k)⌉, checked against description_length's
        // real-valued log₂ C(G, k), a sum of k terms log₂((G − k + i)/i). A rounded
        // quotient moves its log by at most ε/ln 2, and the log's own rounding adds at
        // most ε·|term|. The running sum adds at most k·ε·(total bits). Together that is
        // at most 2·k·ε·(bits + 1), and the band is twice that.
        let rank_width = length - prefix_integer_len_bits(cardinality as u64 + 1).expect("length");
        let real_bits = selection_bits(universe as i64, cardinality as i64);
        let band = 4.0 * cardinality as f64 * f64::EPSILON * (real_bits + 1.0);
        assert!(
            rank_width as f64 >= real_bits - band && (rank_width as f64) - 1.0 < real_bits + band,
            "rank width {rank_width} is not ⌈{real_bits}⌉ within {band}"
        );
        assert!(rank_width > 128, "the fixture must cross the u128 range, got {rank_width} bits");
    }

    #[test]
    fn subset_code_refuses_malformed_subsets_and_codewords() {
        assert!(matches!(
            encode_subset(&mut BitString::new(), 10, &[3, 3]),
            Err(CodecError::InvalidInput(_))
        ));
        assert!(matches!(
            encode_subset(&mut BitString::new(), 10, &[4, 2]),
            Err(CodecError::InvalidInput(_))
        ));
        assert!(matches!(
            encode_subset(&mut BitString::new(), 10, &[2, 10]),
            Err(CodecError::InvalidInput(_))
        ));
        // C(5, 2) = 10 ranks in a 4-bit field: rank 15 is not a codeword.
        let mut out = BitString::new();
        encode_prefix_integer(&mut out, 3).expect("encode");
        out.push_bits(15, 4).expect("bits");
        assert!(matches!(
            decode_support_packet(5, &out),
            Err(CodecError::InvalidCodeword(_))
        ));
        // A valid packet with one trailing bit is not the codeword of one subset.
        let mut packet = BitString::new();
        encode_subset(&mut packet, 5, &[1, 4]).expect("encode");
        assert_eq!(decode_support_packet(5, &packet), Ok(vec![1, 4]));
        packet.push_bit(false);
        assert_eq!(
            decode_support_packet(5, &packet),
            Err(CodecError::TrailingBits { remaining: 1 })
        );
        // A cardinality the message cannot back is refused before the binomial is formed:
        // the rank of a 500 000-subset of 10^6 elements needs at least 500 000 bits, and this
        // packet holds none after its cardinality codeword.
        let mut hostile = BitString::new();
        encode_prefix_integer(&mut hostile, 500_001).expect("encode");
        assert!(matches!(
            decode_support_packet(1_000_000, &hostile),
            Err(CodecError::InvalidCodeword(_))
        ));
    }

    #[test]
    fn ordered_dag_code_round_trips_and_refuses_forward_reads() {
        let nodes = vec![
            DagNode { label: 0, arguments: vec![] },
            DagNode { label: 2, arguments: vec![0, 0] },
            DagNode { label: 1, arguments: vec![1, 0] },
            DagNode { label: 3, arguments: vec![2] },
        ];
        let mut out = BitString::new();
        encode_ordered_dag(&mut out, 4, &nodes).expect("encode");
        // L_int(5) + 4 labels·2 + arities L_int(3)+L_int(3)+L_int(2) + argument widths
        // ⌈log₂ max(i, 2)⌉: 1+1 at node 1, 1+1 at node 2, 2 at node 3.
        let expected = prefix_integer_len_bits(5).expect("length")
            + 8
            + prefix_integer_len_bits(3).expect("length") * 2
            + prefix_integer_len_bits(2).expect("length")
            + 6;
        assert_eq!(out.len_bits(), expected);
        let mut reader = out.reader();
        assert_eq!(decode_ordered_dag(&mut reader, 4), Ok(nodes.clone()));
        assert_eq!(reader.finish(), Ok(()));
        let forward = vec![DagNode { label: 0, arguments: vec![] }, DagNode { label: 0, arguments: vec![1] }];
        assert!(matches!(
            encode_ordered_dag(&mut BitString::new(), 4, &forward),
            Err(CodecError::InvalidInput(_))
        ));
        // A node count far above the message is refused before allocation.
        let mut oversized = BitString::new();
        encode_prefix_integer(&mut oversized, 1 << 40).expect("encode");
        assert!(matches!(
            decode_ordered_dag(&mut oversized.reader(), 4),
            Err(CodecError::InvalidCodeword(_))
        ));
        // The second node can only read the first, yet its arguments spend a bit each, so
        // a message announcing 2^40 of them in a few dozen bits is refused before any
        // argument is allocated (mpd-verify, #2951).
        let mut announced = BitString::new();
        encode_prefix_integer(&mut announced, 3).expect("encode");
        encode_fixed_index(&mut announced, 0, 4).expect("encode");
        encode_fixed_index(&mut announced, 0, 4).expect("encode");
        encode_prefix_integer(&mut announced, (1 << 40) + 1).expect("encode");
        assert!(matches!(
            decode_ordered_dag(&mut announced.reader(), 4),
            Err(CodecError::InvalidCodeword(_))
        ));
        // Positive control: the same header with arity 2 and two one-bit arguments decodes.
        let pair = vec![
            DagNode { label: 0, arguments: vec![] },
            DagNode { label: 0, arguments: vec![0, 0] },
        ];
        let mut small = BitString::new();
        encode_ordered_dag(&mut small, 4, &pair).expect("encode");
        assert_eq!(
            small.len_bits(),
            prefix_integer_len_bits(3).expect("length") * 2 + 4 + 2
        );
        assert_eq!(decode_ordered_dag(&mut small.reader(), 4), Ok(pair));
        // The unused codeword 1 at the second node names no earlier node and is refused.
        let mut forward_read = BitString::new();
        encode_prefix_integer(&mut forward_read, 3).expect("encode");
        encode_fixed_index(&mut forward_read, 0, 4).expect("encode");
        encode_fixed_index(&mut forward_read, 0, 4).expect("encode");
        encode_prefix_integer(&mut forward_read, 2).expect("encode");
        forward_read.push_bit(true);
        assert!(matches!(
            decode_ordered_dag(&mut forward_read.reader(), 4),
            Err(CodecError::InvalidCodeword(_))
        ));
    }

    #[test]
    fn union_library_counts_per_input_and_global_supports_apart() {
        let library = union_support_library(&[vec![3, 9], vec![9, 40], vec![3]]).expect("library");
        assert_eq!(library.components, vec![3, 9, 40]);
        assert_eq!(library.supports, vec![vec![0, 1], vec![1, 2], vec![0]]);
        assert_eq!(library.summed_support, 5);
        assert!(matches!(
            union_support_library(&[vec![9, 3]]),
            Err(CodecError::InvalidInput(_))
        ));
    }

    /// A test artifact whose decoder returns a stored figure, so each fidelity fixture states its
    /// evidence directly and the comparison's refusals are exercised one verdict at a time.
    struct StoredFigure(f64);

    impl DecodableArtifact for StoredFigure {
        type Decoded = f64;

        fn decode(&self) -> Result<f64, String> {
            Ok(self.0)
        }
    }

    fn stored_fidelity(
        value: f64,
        numerical_error: f64,
        tolerance: f64,
    ) -> DecodedFidelity<(), &'static str> {
        decode_then_evaluate(
            &StoredFigure(value),
            |decoded: &f64| Ok(*decoded),
            &0.0,
            |output: &f64, reference: &f64| {
                EvidenceStatus::exact(
                    (output - reference).abs(),
                    numerical_error,
                    ExactBasis::Algebraic,
                    None,
                    "stored figure",
                )
                .map_err(|error| error.to_string())
            },
            tolerance,
        )
        .expect("a finite figure under a finite tolerance")
    }

    #[test]
    fn code_comparison_refuses_artifacts_that_do_not_meet_one_tolerance() {
        let short = stored_fidelity(0.0, 0.0, 0.25);
        let long = stored_fidelity(0.2, 1e-15, 0.25);
        assert_eq!(short.verdict(), FidelityVerdict::Meets);
        assert_eq!(long.verdict(), FidelityVerdict::Meets);
        assert_eq!(code_saving_at_proven_fidelity((300, &long), (10, &short)), Ok(290));
        // A proven violation is refused.
        let violating = stored_fidelity(0.3, 0.0, 0.25);
        assert_eq!(violating.verdict(), FidelityVerdict::Violates);
        assert!(code_saving_at_proven_fidelity((300, &long), (10, &violating)).is_err());
        // A figure within its rounding band of the tolerance proves neither side and is refused.
        let unresolved = stored_fidelity(0.25, 1e-16, 0.25);
        assert_eq!(unresolved.verdict(), FidelityVerdict::Unresolved);
        assert!(code_saving_at_proven_fidelity((300, &long), (10, &unresolved)).is_err());
        // Two artifacts that each meet their own tolerance are still not one comparison when the
        // tolerances differ, here by one unit in the last place.
        let looser = stored_fidelity(0.0, 0.0, f64::from_bits(0.25_f64.to_bits() + 1));
        assert_eq!(looser.verdict(), FidelityVerdict::Meets);
        assert!(code_saving_at_proven_fidelity((300, &long), (10, &looser)).is_err());
    }

    // The A8 fixture's label alphabet: an input read, the native identity primitive,
    // one instance of the projector family, and a sum masked by the input's packet.
    const INPUT: usize = 0;
    const IDENTITY: usize = 1;
    const FAMILY_INSTANCE: usize = 2;
    const MASKED_SUM: usize = 3;
    const LABELS: usize = 4;
    // The declared resolution of each projector label on RP^1, in bits.
    const LABEL_RESOLUTION_BITS: u32 = 8;

    /// `Σ_{c∈S} (2/C)(v(t_c)·x) v(t_c)` over the decoded labels `t_c`, with `C` labels.
    fn projector_output(x: [f64; 2], support: &[usize], labels: &[f64]) -> [f64; 2] {
        let weight_scale = 2.0 / labels.len() as f64;
        let mut output = [0.0_f64; 2];
        for &instance in support {
            let (sine, cosine) = labels[instance].sin_cos();
            let weight = weight_scale * (cosine * x[0] + sine * x[1]);
            output[0] += weight * cosine;
            output[1] += weight * sine;
        }
        output
    }

    fn distance(output: [f64; 2], x: [f64; 2]) -> f64 {
        ((x[0] - output[0]).powi(2) + (x[1] - output[1]).powi(2)).sqrt()
    }

    /// A derived bound on the rounding error of one executed distortion over `instances` labels.
    ///
    /// Every partial sum and the output stay within norm 2 (`‖P_S x‖ ≤ 2|S|/C ≤ 2`). Each output
    /// coordinate is at most `2C + 12` rounded operations from the decoded labels and the input
    /// (direction, dot product, scale, accumulate, subtract, norm), so its relative error is at
    /// most Higham's `γ_m = m·u/(1 − m·u)` of that count ([`accumulation_growth`], `u = ε/2`), and
    /// the distortion, a norm of magnitude at most 3, moves by at most `3·γ_m`.
    fn distortion_roundoff(instances: usize) -> f64 {
        3.0 * accumulation_growth(2 * instances + 12)
    }

    /// What an A8 artifact decodes to: the native identity, or a projector family with its
    /// decoded labels and each input's decoded support.
    enum DecodedProgram {
        Identity,
        ProjectorFamily {
            labels: Vec<f64>,
            supports: Vec<Vec<usize>>,
        },
    }

    impl DecodedProgram {
        fn family_parts(&self) -> Option<(&[f64], &[Vec<usize>])> {
            match self {
                Self::ProjectorFamily { labels, supports } => {
                    Some((labels.as_slice(), supports.as_slice()))
                }
                Self::Identity => None,
            }
        }
    }

    /// An A8 artifact as transmitted: the library (the program graph, then a projector family's
    /// label message) and one packet per input.
    struct A8Artifact(LibraryPacketArtifact);

    impl DecodableArtifact for A8Artifact {
        type Decoded = DecodedProgram;

        fn decode(&self) -> Result<DecodedProgram, String> {
            let mut reader = self.0.library.reader();
            let nodes = decode_ordered_dag(&mut reader, LABELS).map_err(|error| error.to_string())?;
            let instances = nodes.iter().filter(|node| node.label == FAMILY_INSTANCE).count();
            if instances == 0 {
                reader.finish().map_err(|error| error.to_string())?;
                if self.0.packets.iter().any(|packet| !packet.is_empty()) {
                    return Err("the identity program reads no packet".to_string());
                }
                return Ok(DecodedProgram::Identity);
            }
            let quotient = PeriodicQuotient::new(std::f64::consts::PI)?;
            let labels = QuotientCode::read(&mut reader, quotient)?.decode()?;
            reader.finish().map_err(|error| error.to_string())?;
            if labels.len() != instances {
                return Err(format!("{} labels for {instances} instances", labels.len()));
            }
            let supports = self
                .0
                .packets
                .iter()
                .map(|packet| {
                    decode_support_packet(instances, packet).map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(DecodedProgram::ProjectorFamily { labels, supports })
        }
    }

    /// Execute a decoded A8 program on every input.
    fn execute(program: &DecodedProgram, inputs: &[[f64; 2]]) -> Result<Vec<[f64; 2]>, String> {
        match program {
            DecodedProgram::Identity => Ok(inputs.to_vec()),
            DecodedProgram::ProjectorFamily { labels, supports } => {
                if supports.len() != inputs.len() {
                    return Err(format!(
                        "{} packets for {} inputs",
                        supports.len(),
                        inputs.len()
                    ));
                }
                Ok(inputs
                    .iter()
                    .zip(supports)
                    .map(|(&x, support)| projector_output(x, support, labels))
                    .collect())
            }
        }
    }

    /// The declared distortion of executed outputs against the native reference: the largest
    /// `‖y − x‖` over every executed input, exhaustive, with its derived rounding bound and the
    /// worst input as witness.
    fn worst_distortion(
        outputs: &Vec<[f64; 2]>,
        reference: &Vec<[f64; 2]>,
        instances: usize,
    ) -> Result<EvidenceStatus<usize, &'static str>, String> {
        if outputs.len() != reference.len() || outputs.is_empty() {
            return Err(format!(
                "{} outputs for {} reference inputs",
                outputs.len(),
                reference.len()
            ));
        }
        let (worst, value) = outputs
            .iter()
            .zip(reference)
            .map(|(&output, &x)| distance(output, x))
            .enumerate()
            .fold((0_usize, 0.0_f64), |(worst, largest), (input, distortion)| {
                if distortion > largest {
                    (input, distortion)
                } else {
                    (worst, largest)
                }
            });
        EvidenceStatus::exact(
            value,
            distortion_roundoff(instances),
            ExactBasis::Exhaustive {
                cardinality: outputs.len() as u64,
            },
            Some(worst),
            "the declared unit inputs",
        )
        .map_err(|error| error.to_string())
    }

    #[test]
    fn identity_program_is_shorter_than_projector_family_at_equal_fidelity_a8() {
        let instances = 16_usize;
        let tolerance = 0.25_f64;
        let inputs: Vec<[f64; 2]> = (0..24)
            .map(|index| {
                let angle = 2.0 * std::f64::consts::PI * (index as f64 + 1.0 / 3.0) / 24.0;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let roundoff = distortion_roundoff(instances);

        // Identity program: read the input, apply the native identity. No packet.
        let identity_nodes = vec![
            DagNode { label: INPUT, arguments: vec![] },
            DagNode { label: IDENTITY, arguments: vec![0] },
        ];
        let mut identity_library = BitString::new();
        encode_ordered_dag(&mut identity_library, LABELS, &identity_nodes).expect("identity library");
        // Hand-derived: node count L_int(3) = 3, two labels at 2 bits, the second node's
        // arity L_int(2) = 3, and its one argument at ⌈log₂ max(1, 2)⌉ = 1 bit.
        assert_eq!(identity_library.len_bits(), 11);
        let identity = A8Artifact(LibraryPacketArtifact {
            library: identity_library,
            packets: vec![BitString::new(); inputs.len()],
        });
        let identity_fidelity = decode_then_evaluate(
            &identity,
            |program: &DecodedProgram| execute(program, &inputs),
            &inputs,
            |outputs: &Vec<[f64; 2]>, reference: &Vec<[f64; 2]>| {
                worst_distortion(outputs, reference, instances)
            },
            tolerance,
        )
        .expect("the identity artifact decodes and executes");
        assert_eq!(identity_fidelity.verdict(), FidelityVerdict::Meets);

        // Projector family: the program graph, then the labels t_c = πc/C sent once on RP^1 at
        // the declared resolution through precision.rs. The decoder executes the decoded labels.
        let mut family_nodes = vec![DagNode { label: INPUT, arguments: vec![] }];
        family_nodes.extend((0..instances).map(|_| DagNode { label: FAMILY_INSTANCE, arguments: vec![0] }));
        family_nodes.push(DagNode { label: MASKED_SUM, arguments: (1..=instances).collect() });
        let mut family_library = BitString::new();
        encode_ordered_dag(&mut family_library, LABELS, &family_nodes).expect("family graph");
        // Hand-derived: node count L_int(19) = 11, 18 labels at 2 bits = 36, instance arities
        // 16·L_int(2) = 48, instance arguments Σ_{i=1..16}⌈log₂ max(i, 2)⌉ = 50, and the
        // masked sum's L_int(17) = 11 plus 16 arguments at ⌈log₂ 17⌉ = 5 bits = 80.
        assert_eq!(family_library.len_bits(), 236);
        let true_labels: Vec<f64> = (0..instances)
            .map(|instance| std::f64::consts::PI * instance as f64 / instances as f64)
            .collect();
        let label_code = QuotientCode::encode(
            &true_labels,
            PeriodicQuotient::new(std::f64::consts::PI).expect("the RP^1 period"),
            LABEL_RESOLUTION_BITS,
        )
        .expect("the labels encode");
        label_code.write(&mut family_library).expect("the labels write");
        // Hand-derived label message: count L_int(17) = 11, resolution L_int(9) = 7, and 16
        // indices at 8 bits = 128.
        assert_eq!(family_library.len_bits(), 236 + 146);
        let decoded_labels = label_code.decode().expect("the labels decode");
        for (&decoded_label, &true_label) in decoded_labels.iter().zip(&true_labels) {
            assert!(
                (decoded_label - true_label).abs() <= label_code.worst_case_error(),
                "label {true_label} decoded to {decoded_label}"
            );
        }

        // Per input, the fewest largest-overlap instances whose decoded output meets the
        // tolerance with its roundoff counted, rounded up the way `EvidenceStatus::certifies_at_most`
        // rounds, so the executed figures below reach the same verdict.
        let supports: Vec<Vec<usize>> = inputs
            .iter()
            .map(|&x| {
                let mut order: Vec<usize> = (0..instances).collect();
                order.sort_by(|&left, &right| {
                    let overlap = |instance: usize| {
                        let (sine, cosine) = decoded_labels[instance].sin_cos();
                        (cosine * x[0] + sine * x[1]).powi(2)
                    };
                    overlap(right).total_cmp(&overlap(left))
                });
                (1..=instances)
                    .map(|count| {
                        let mut support = order[..count].to_vec();
                        support.sort_unstable();
                        support
                    })
                    .find(|support| {
                        (distance(projector_output(x, support, &decoded_labels), x) + roundoff)
                            .next_up()
                            <= tolerance
                    })
                    .expect("the all-on sum is the identity to rounding, so some support meets the tolerance")
            })
            .collect();
        let library = union_support_library(&supports).expect("union library");
        assert_eq!(library.components, (0..instances).collect::<Vec<_>>());
        // P18: the per-input count E_x|S_x| is not the library size |∪_x S_x|.
        assert!(
            library.summed_support < (inputs.len() * library.components.len()) as u64,
            "mean support {} must be below the library size {}",
            library.summed_support as f64 / inputs.len() as f64,
            library.components.len()
        );
        let family = A8Artifact(LibraryPacketArtifact {
            library: family_library,
            packets: encode_support_packets(instances, &library.supports).expect("packets"),
        });

        // The decoder rebuilds the labels and every packet from the artifact alone.
        let decoded = family.decode().expect("the family artifact decodes");
        let (labels, decoded_supports) = decoded
            .family_parts()
            .expect("a library with family instances decodes to a projector family");
        assert_eq!(labels, decoded_labels.as_slice());
        let minimum_cardinality = (instances as f64 * (1.0 - tolerance) / 2.0).ceil() as usize;
        for (input, support) in decoded_supports.iter().enumerate() {
            assert_eq!(support, &library.supports[input]);
            // P11: error ε on a unit input needs |S| ≥ C(1 − ε)/2.
            assert!(
                support.len() >= minimum_cardinality,
                "input {input} met the tolerance with {} of {instances} instances, below the P11 bound {minimum_cardinality}",
                support.len()
            );
        }
        let family_fidelity = decode_then_evaluate(
            &family,
            |program: &DecodedProgram| execute(program, &inputs),
            &inputs,
            |outputs: &Vec<[f64; 2]>, reference: &Vec<[f64; 2]>| {
                worst_distortion(outputs, reference, instances)
            },
            tolerance,
        )
        .expect("the family artifact decodes and executes");
        assert_eq!(family_fidelity.verdict(), FidelityVerdict::Meets);

        let saving = code_saving_at_proven_fidelity(
            (family.0.total_bits(), &family_fidelity),
            (identity.0.total_bits(), &identity_fidelity),
        )
        .expect("both artifacts meet one declared tolerance");
        assert!(
            saving > 0,
            "identity ({} bits) must be shorter than the projector family ({} bits)",
            identity.0.total_bits(),
            family.0.total_bits()
        );
        // A bound free of the packet choice: every faithful packet costs at least the least
        // subset code over the admissible cardinalities.
        let least_packet = (minimum_cardinality..=instances)
            .map(|cardinality| subset_code_len_bits(instances, cardinality).expect("length"))
            .min()
            .expect("admissible cardinalities");
        let family_floor = family.0.library.len_bits() + inputs.len() as u64 * least_packet;
        assert!(family.0.total_bits() >= family_floor);
        assert!(
            identity.0.total_bits() < family_floor,
            "identity ({} bits) must beat every faithful projector artifact (at least {family_floor} bits)",
            identity.0.total_bits()
        );

        // Positive control: single-instance packets leave ‖x − P_c x‖ ≥ 1 − 2/C = 0.875 on every
        // input, a proven violation, and the comparison refuses them instead of reporting their
        // shorter packets.
        let single: Vec<Vec<usize>> = supports.iter().map(|support| vec![support[0]]).collect();
        let unfaithful = A8Artifact(LibraryPacketArtifact {
            library: family.0.library.clone(),
            packets: encode_support_packets(instances, &single).expect("packets"),
        });
        let unfaithful_fidelity = decode_then_evaluate(
            &unfaithful,
            |program: &DecodedProgram| execute(program, &inputs),
            &inputs,
            |outputs: &Vec<[f64; 2]>, reference: &Vec<[f64; 2]>| {
                worst_distortion(outputs, reference, instances)
            },
            tolerance,
        )
        .expect("the single-instance artifact decodes and executes");
        assert_eq!(unfaithful_fidelity.verdict(), FidelityVerdict::Violates);
        assert!(code_saving_at_proven_fidelity(
            (unfaithful.0.total_bits(), &unfaithful_fidelity),
            (identity.0.total_bits(), &identity_fidelity),
        )
        .is_err());
    }

    // The P12 fixture's declared fidelity tolerance: a certified risk of 0 meets it and a refuted
    // risk of 1 does not.
    const CORE_TOLERANCE: f64 = 0.5;

    /// A separation oracle whose risk is zero exactly when every `core` component is kept. A
    /// refuting witness turns off every component outside the candidate support.
    struct CoreOracle {
        components: usize,
        core: Vec<usize>,
    }

    impl SeparationOracle for CoreOracle {
        type Mask = Vec<f64>;
        type Domain = &'static str;
        type Error = EvidenceStatusError;

        fn components(&self) -> usize {
            self.components
        }

        fn perturbed_components(&self, mask: &Vec<f64>) -> Vec<usize> {
            (0..mask.len())
                .filter(|&component| mask[component] != 1.0)
                .collect()
        }

        fn separate(
            &mut self,
            support: &ComponentSet,
        ) -> Result<EvidenceStatus<Vec<f64>, &'static str>, EvidenceStatusError> {
            let kept = |component: usize| support.members().binary_search(&component).is_ok();
            if self.core.iter().all(|&component| kept(component)) {
                EvidenceStatus::exact(0.0, 0.0, ExactBasis::Algebraic, None, "core kept")
            } else {
                let mask = (0..self.components)
                    .map(|component| if kept(component) { 1.0 } else { 0.0 })
                    .collect();
                EvidenceStatus::counterexample(1.0, 0.0, CORE_TOLERANCE, mask)
            }
        }

        fn evaluate(
            &mut self,
            mask: &Vec<f64>,
        ) -> Result<EvidenceStatus<Vec<f64>, &'static str>, EvidenceStatusError> {
            let violated = self.core.iter().any(|&component| mask[component] != 1.0);
            EvidenceStatus::exact(
                if violated { 1.0 } else { 0.0 },
                0.0,
                ExactBasis::Exhaustive { cardinality: 1 },
                Some(mask.clone()),
                "one mask",
            )
        }
    }

    /// A registry with 12 reads of one storage `w`, so its use-site ids sort out of ordinal
    /// order (`w#10` precedes `w#2`), a transposed tied read through the alias `w_tied`, a
    /// second matrix `v` with one read, a norm gain `g` read as stored values, and a matrix
    /// `u` nothing reads. `extra_storage` registers one more tensor, which changes the
    /// fingerprint.
    fn address_registry(extra_storage: bool) -> TensorRegistry {
        let id = |name: &str| TensorId(name.to_string());
        let mut shapes = vec![("g", vec![3]), ("u", vec![3, 2]), ("v", vec![2, 2]), ("w", vec![2, 3])];
        if extra_storage {
            shapes.push(("x", vec![1, 1]));
        }
        let mut registry = TensorRegistry::default();
        for (name, shape) in shapes {
            registry
                .register_storage(id(name), ArrayD::<f64>::zeros(shape).view())
                .expect("the storage tensor registers");
        }
        registry.register_alias(id("w_tied"), id("w")).expect("the alias registers");
        let reads = (0..12)
            .map(|ordinal| (UseSiteId::read(&id("w"), ordinal), id("w"), UseMap::Linear(TieOrientation::Identity)))
            .chain([
                (UseSiteId::read(&id("w_tied"), 0), id("w_tied"), UseMap::Linear(TieOrientation::Transpose)),
                (UseSiteId::read(&id("v"), 0), id("v"), UseMap::Linear(TieOrientation::Identity)),
                (UseSiteId::read(&id("g"), 0), id("g"), UseMap::Stored),
            ]);
        for (site, reads_name, map) in reads {
            registry.register_use_site(site, reads_name, map).expect("the use site registers");
        }
        registry
    }

    #[test]
    fn edit_scope_address_indexes_the_registry_id_order_not_the_use_ordinal() {
        let registry = address_registry(false);
        let id = |name: &str| TensorId(name.to_string());
        let (v, w) = (id("v"), id("w"));
        // u is read by nothing and g is not a matrix, so the editable storage is [v, w]: 1 bit.
        // w has 13 uses in id order (w#0, w#1, w#10, w#11, w#2, ..., w#9, w_tied#0), so its scope
        // alphabet has 14 symbols in 4 bits, and w#2 is sent as 1 + its position 4, not 1 + its
        // ordinal 2.
        let mut message = BitString::new();
        encode_edit_scope(&mut message, &registry, &EditScope::UseSite(UseSiteId::read(&w, 2)))
            .expect("w#2 encodes");
        let mut reader = message.reader();
        assert_eq!(reader.read_bits(1).expect("the storage index"), 1);
        assert_eq!(reader.read_bits(4).expect("the scope index"), 5);
        reader.finish().expect("the scope is 5 bits");

        let mut scopes = vec![EditScope::Global(v.clone()), EditScope::Global(w.clone())];
        scopes.extend(
            registry.use_sites_of(&v).into_iter().chain(registry.use_sites_of(&w)).cloned().map(EditScope::UseSite),
        );
        assert_eq!(scopes.len(), 16);
        for scope in &scopes {
            let mut message = BitString::new();
            encode_edit_scope(&mut message, &registry, scope).expect("the scope encodes");
            assert_eq!(message.len_bits(), edit_scope_len_bits(&registry, scope).expect("the length"));
            let mut reader = message.reader();
            assert_eq!(&decode_edit_scope(&mut reader, &registry).expect("the scope reads back"), scope);
            reader.finish().expect("no trailing bits");
        }

        // occurrence refuses an alias named as storage, a non-matrix storage, an unused tensor and
        // an unknown use site, so none of them has a codeword.
        for refused in [
            EditScope::Global(id("w_tied")),
            EditScope::Global(id("g")),
            EditScope::Global(id("u")),
            EditScope::UseSite(UseSiteId("w#99".to_string())),
        ] {
            assert!(
                matches!(encode_edit_scope(&mut BitString::new(), &registry, &refused), Err(CodecError::InvalidInput(_))),
                "{refused:?} must be refused"
            );
        }
        // A hostile message: storage w, then scope index 15 of a 14-symbol alphabet.
        let mut hostile = BitString::new();
        hostile.push_bits(1, 1).expect("the storage bit");
        hostile.push_bits(15, 4).expect("the scope bits");
        assert!(matches!(
            decode_edit_scope(&mut hostile.reader(), &registry),
            Err(CodecError::InvalidCodeword(_))
        ));
    }

    #[test]
    fn address_header_names_one_teacher_and_declares_the_position_universe() {
        let registry = address_registry(false);
        let mut message = BitString::new();
        encode_address_header(&mut message, &registry, 12).expect("the header encodes");
        assert_eq!(message.len_bits(), address_header_len_bits(12).expect("the length"));
        let mut reader = message.reader();
        let header = decode_address_header(&mut reader, &registry).expect("the header reads back");
        reader.finish().expect("no trailing bits");
        assert_eq!(header, (registry.teacher_fingerprint(), 12));
        // Positive control: one more storage tensor changes the fingerprint, and the header written
        // for the first registry is refused against the second.
        let other = address_registry(true);
        assert_ne!(other.teacher_fingerprint(), registry.teacher_fingerprint());
        assert!(matches!(
            decode_address_header(&mut message.reader(), &other),
            Err(CodecError::InvalidCodeword(_))
        ));
        assert!(matches!(
            encode_address_header(&mut BitString::new(), &registry, 0),
            Err(CodecError::InvalidInput(_))
        ));
        assert!(address_header_len_bits(0).is_err());
    }

    #[test]
    fn position_scope_code_sends_every_as_one_bit_and_a_declared_set_as_its_subset_code() {
        let length = 12;
        // declared {3, 7}: 1 flag bit + L_int(3) = 3 + ceil(log2 C(12, 2)) = ceil(log2 66) = 7.
        let pair = PositionScope::declared(vec![3, 7]).expect("the positions are increasing");
        // A key-value cache filled before the edit keeps positions 0..5 clean.
        let cached_prefix = PositionScope::declared((5..length).collect()).expect("the positions are increasing");
        let cached_bits = 1 + subset_code_len_bits(length, 7).expect("the subset length");
        for (scope, bits) in [(PositionScope::every(), 1), (pair, 1 + 3 + 7), (cached_prefix, cached_bits)] {
            let mut message = BitString::new();
            encode_position_scope(&mut message, &scope, length).expect("the scope encodes");
            assert_eq!(message.len_bits(), bits);
            assert_eq!(position_scope_len_bits(&scope, length).expect("the length"), bits);
            let mut reader = message.reader();
            assert_eq!(decode_position_scope(&mut reader, length).expect("the scope reads back"), scope);
            reader.finish().expect("no trailing bits");
        }
        let outside = PositionScope::declared(vec![11, 12]).expect("the positions are increasing");
        assert!(matches!(
            encode_position_scope(&mut BitString::new(), &outside, length),
            Err(CodecError::InvalidInput(_))
        ));
        // A hostile message: the declared flag followed by the empty subset, which
        // PositionScope::declared refuses.
        let mut hostile = BitString::new();
        hostile.push_bit(true);
        encode_subset(&mut hostile, length, &[]).expect("the empty subset has a codeword");
        assert!(matches!(
            decode_position_scope(&mut hostile.reader(), length),
            Err(CodecError::InvalidCodeword(_))
        ));
    }

    #[test]
    fn edit_address_round_trips_a_record_and_refuses_one_checked_against_another_registry() {
        let registry = address_registry(false);
        let w = TensorId("w".to_string());
        let length = 12;
        let delta = FactoredEdit::new(Array2::from_elem((2, 1), 0.5), Array2::from_elem((3, 1), -0.25))
            .expect("the factors agree on the term count");
        let record = ParameterEditRecord::new(
            &registry,
            EditScope::UseSite(UseSiteId::read(&w, 10)),
            PositionScope::declared((5..length).collect()).expect("the positions are increasing"),
            delta.clone(),
        )
        .expect("the record is checked against the registry");
        let mut message = BitString::new();
        encode_address_header(&mut message, &registry, length).expect("the header encodes");
        encode_edit_address(&mut message, &registry, length, &record).expect("the address encodes");
        assert_eq!(
            message.len_bits(),
            address_header_len_bits(length).expect("the header length")
                + edit_scope_len_bits(&registry, record.scope()).expect("the scope length")
                + position_scope_len_bits(record.positions(), length).expect("the positions length")
        );
        let mut reader = message.reader();
        let (_, decoded_length) = decode_address_header(&mut reader, &registry).expect("the header reads back");
        let (scope, positions) =
            decode_edit_address(&mut reader, &registry, decoded_length).expect("the address reads back");
        reader.finish().expect("no trailing bits");
        assert_eq!((&scope, &positions), (record.scope(), record.positions()));

        // A record checked against another teacher's registry numbers its uses on another forward
        // path, so it is refused. Positive control: against its own registry it encodes.
        let other = address_registry(true);
        let foreign = ParameterEditRecord::new(&other, EditScope::Global(w), PositionScope::every(), delta)
            .expect("the record is checked against the other registry");
        assert!(matches!(
            encode_edit_address(&mut BitString::new(), &registry, length, &foreign),
            Err(CodecError::InvalidInput(_))
        ));
        assert!(encode_edit_address(&mut BitString::new(), &other, length, &foreign).is_ok());
    }

    #[test]
    fn enumerative_subset_code_drives_the_p12_minimum_code_search() {
        // Hand-derived from L_int(k + 1) = 1, 3, 3, 6, 6, 6, 6, 7, 7 and
        // ⌈log₂ C(8, k)⌉ = 0, 3, 5, 6, 7, 6, 5, 3, 0 for k = 0..=8.
        let lengths = [1_u64, 6, 8, 12, 13, 12, 11, 10, 7];
        for (size, &bits) in lengths.iter().enumerate() {
            assert_eq!(EnumerativeSubsetCode.support_bits(8, size), Ok(bits), "L(8, {size})");
        }
        assert!(matches!(
            EnumerativeSubsetCode.support_bits(8, 9),
            Err(CodecError::InvalidInput(_))
        ));

        let mut oracle = CoreOracle {
            components: 8,
            core: vec![0, 1, 2],
        };
        let search = minimum_code_support(
            &mut oracle,
            &EnumerativeSubsetCode,
            CORE_TOLERANCE,
            FailureHypergraph::new(8),
        )
        .expect("the full support keeps the core, so the search closes");
        // The empty candidate (1 bit) is refuted. At hitting-set size 1 the cheapest admissible
        // size is 1 (6 bits), and all 8 singletons miss the core. At size 2 the cheapest is the
        // full support (7 bits, below L(8, 2) = 8), which is certified: 1 + 8 + 1 separations,
        // whichever singleton the hitting-set solver picks first.
        assert_eq!(search.code.upper_bound(), Some(7.0));
        assert_eq!(search.code.lower_bound(), Some(7.0));
        assert_eq!(search.separations, 10);
        assert_eq!(
            search.certified.map(|found| found.support),
            Some(ComponentSet::all(8))
        );
        // The figure is the true minimum over sufficient supports: each holds the 3-member core,
        // and the least code over sizes 3..=8 is the full support's.
        let least_sufficient = (3..=8)
            .map(|size| EnumerativeSubsetCode.support_bits(8, size).expect("length"))
            .min();
        assert_eq!(least_sufficient, Some(7));
        // The size-minimal sufficient support, the core itself, costs 12 bits under this code.
        assert_eq!(EnumerativeSubsetCode.support_bits(8, 3), Ok(12));
    }
}
