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
