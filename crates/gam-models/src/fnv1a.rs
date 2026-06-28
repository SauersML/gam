//! Shared FNV-1a 64-bit hashing primitive for family-local cache
//! fingerprints.
//!
//! Family implementations use this to key warm-start and exact-evaluation
//! caches from byte-identical coefficient slices, row states, and scalar
//! parameters. The type gives those caches one canonical byte-feeding
//! convention so distinct cache streams stay mutually consistent.
//!
//! At 64 bits, false collisions across distinct inputs are astronomically
//! rare; on a miss the caller simply re-solves or rebuilds from the uncached
//! path, so the hash is a performance optimization rather than a correctness
//! dependency.

use ndarray::Array1;

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Streaming FNV-1a 64-bit hasher with the byte-feeding conventions shared by
/// family cache fingerprints.
#[derive(Clone, Copy)]
pub(crate) struct Fnv1a {
    hash: u64,
}

impl Fnv1a {
    /// Start a fresh hash seeded with the FNV-1a offset basis.
    #[inline]
    pub(crate) fn new() -> Self {
        Self { hash: FNV_OFFSET }
    }

    /// Mix a single byte (the FNV-1a core step). Used for domain separators
    /// and per-field markers.
    #[inline]
    pub(crate) fn mix_byte(&mut self, byte: u8) {
        self.hash ^= byte as u64;
        self.hash = self.hash.wrapping_mul(FNV_PRIME);
    }

    /// Mix a finite scalar by its IEEE-754 bit pattern, canonicalizing `-0.0`
    /// to `+0.0` so numerically equal scalars hash equal.
    #[inline]
    pub(crate) fn mix_f64(&mut self, x: f64) {
        let bits = if x == 0.0 { 0u64 } else { x.to_bits() };
        for b in bits.to_le_bytes() {
            self.mix_byte(b);
        }
    }

    /// Mix an optional coefficient slice tagged by `marker`. `None` feeds a
    /// distinguished `0xff` sentinel; `Some` feeds a length prefix followed by
    /// each element's canonicalized bits (via [`Fnv1a::mix_f64`]).
    #[inline]
    pub(crate) fn mix_opt_beta(&mut self, marker: u8, beta: Option<&Array1<f64>>) {
        self.mix_byte(marker);
        match beta {
            None => self.mix_byte(0xff),
            Some(v) => {
                let len = v.len() as u64;
                for b in len.to_le_bytes() {
                    self.mix_byte(b);
                }
                for x in v.iter() {
                    self.mix_f64(*x);
                }
            }
        }
    }

    /// Finalize the hash, remapping `0` to `1` so a zero result can serve as a
    /// "never written" cache sentinel without colliding with a real key.
    #[inline]
    pub(crate) fn finish_nonzero(self) -> u64 {
        if self.hash == 0 { 1 } else { self.hash }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A fresh hasher starts at the FNV-1a offset basis.
    #[test]
    fn new_starts_at_fnv_offset_basis() {
        let h = Fnv1a::new();
        assert_eq!(h.hash, 0xcbf2_9ce4_8422_2325u64, "initial hash is FNV offset basis");
    }

    /// `finish_nonzero` returns the hash unchanged when nonzero.
    #[test]
    fn finish_nonzero_nonzero_hash_unchanged() {
        let mut h = Fnv1a::new();
        h.mix_byte(0x42);
        let result = h.finish_nonzero();
        assert_ne!(result, 0, "should not be zero");
        assert_eq!(result, h.hash, "nonzero hash passes through");
    }

    /// `finish_nonzero` returns 1 when the hash is somehow 0.
    #[test]
    fn finish_nonzero_zero_hash_becomes_one() {
        let h = Fnv1a { hash: 0 };
        assert_eq!(h.finish_nonzero(), 1, "zero hash maps to 1");
    }

    /// Same byte sequence always produces the same hash (determinism).
    #[test]
    fn same_bytes_produce_same_hash() {
        let mut a = Fnv1a::new();
        let mut b = Fnv1a::new();
        for byte in [0x01u8, 0x02, 0xFF, 0xAB] {
            a.mix_byte(byte);
            b.mix_byte(byte);
        }
        assert_eq!(a.finish_nonzero(), b.finish_nonzero(), "deterministic");
    }

    /// Different byte sequences produce different hashes (collision avoidance
    /// for simple cases — not a general guarantee, but covers the basics).
    #[test]
    fn different_bytes_produce_different_hashes() {
        let mut a = Fnv1a::new();
        a.mix_byte(0x01);
        let mut b = Fnv1a::new();
        b.mix_byte(0x02);
        assert_ne!(a.finish_nonzero(), b.finish_nonzero(), "distinct bytes → distinct hashes");
    }

    /// `mix_f64` canonicalizes -0.0 to +0.0 so they hash identically.
    #[test]
    fn mix_f64_negative_zero_equals_positive_zero() {
        let mut a = Fnv1a::new();
        a.mix_f64(-0.0_f64);
        let mut b = Fnv1a::new();
        b.mix_f64(0.0_f64);
        assert_eq!(a.finish_nonzero(), b.finish_nonzero(), "-0.0 and +0.0 should hash equally");
    }

    /// `mix_f64` distinguishes distinct nonzero values.
    #[test]
    fn mix_f64_distinct_values_differ() {
        let mut a = Fnv1a::new();
        a.mix_f64(1.0_f64);
        let mut b = Fnv1a::new();
        b.mix_f64(2.0_f64);
        assert_ne!(a.finish_nonzero(), b.finish_nonzero(), "1.0 and 2.0 hash differently");
    }

    /// `mix_opt_beta(Some([]))` and `mix_opt_beta(None)` produce different hashes.
    #[test]
    fn mix_opt_beta_none_differs_from_empty_some() {
        let mut a = Fnv1a::new();
        a.mix_opt_beta(0, None);
        let mut b = Fnv1a::new();
        b.mix_opt_beta(0, Some(&array![]));
        assert_ne!(a.finish_nonzero(), b.finish_nonzero(), "None vs Some([]) must differ");
    }

    /// `mix_opt_beta` with two different arrays produces different hashes.
    #[test]
    fn mix_opt_beta_different_arrays_differ() {
        let v1 = array![1.0_f64, 2.0, 3.0];
        let v2 = array![1.0_f64, 2.0, 4.0];
        let mut a = Fnv1a::new();
        a.mix_opt_beta(0, Some(&v1));
        let mut b = Fnv1a::new();
        b.mix_opt_beta(0, Some(&v2));
        assert_ne!(a.finish_nonzero(), b.finish_nonzero(), "distinct arrays → distinct hashes");
    }

    /// `mix_opt_beta` with the same array produces the same hash (determinism).
    #[test]
    fn mix_opt_beta_same_array_deterministic() {
        let v = array![3.14_f64, -2.71, 0.0];
        let mut a = Fnv1a::new();
        a.mix_opt_beta(1, Some(&v));
        let mut b = Fnv1a::new();
        b.mix_opt_beta(1, Some(&v));
        assert_eq!(a.finish_nonzero(), b.finish_nonzero(), "same array → same hash");
    }
}
