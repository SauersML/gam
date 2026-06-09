//! Fingerprint keying for the warm-start cache.
//!
//! A [`Fingerprint`] is a SHA-256 hash; two fits whose (data, spec) byte
//! representations agree under [`Fingerprinter`] absorption produce the same
//! key. Adversarial collisions don't matter — per-variant warm-start
//! validators are the correctness fail-safe; the fingerprint is just a fast
//! filter.

use sha2::{Digest, Sha256};
use std::fmt;

/// 256-bit cache key.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fingerprint([u8; 32]);

impl Fingerprint {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in &self.0 {
            use std::fmt::Write;
            write!(&mut s, "{:02x}", b).expect("writing to String is infallible");
        }
        s
    }

    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() != 64 {
            return None;
        }
        let bytes = s.as_bytes();
        let mut out = [0u8; 32];
        for i in 0..32 {
            let hi = from_hex_nibble(bytes[2 * i])?;
            let lo = from_hex_nibble(bytes[2 * i + 1])?;
            out[i] = (hi << 4) | lo;
        }
        Some(Fingerprint(out))
    }
}

const fn from_hex_nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

impl fmt::Debug for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fingerprint({})", self.to_hex())
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

/// Streaming hasher for building a [`Fingerprint`].
///
/// Each `absorb_*` writes a short type-tag + length before the data, so
/// `absorb_f64(b"x", 0.5)` cannot collide with `absorb_bytes(b"x", <the 8
/// little-endian bytes of 0.5>)`.
pub struct Fingerprinter {
    h: Sha256,
}

impl Fingerprinter {
    pub fn new() -> Self {
        Self { h: Sha256::new() }
    }

    /// Absorb a tag with no payload. Useful for structural separators.
    pub fn absorb_tag(&mut self, tag: &[u8]) {
        self.h.update((tag.len() as u32).to_le_bytes());
        self.h.update(tag);
    }

    pub fn absorb_bytes(&mut self, tag: &[u8], data: &[u8]) {
        self.absorb_tag(tag);
        self.h.update((data.len() as u64).to_le_bytes());
        self.h.update(data);
    }

    pub fn absorb_str(&mut self, tag: &[u8], s: &str) {
        self.absorb_bytes(tag, s.as_bytes());
    }

    pub fn absorb_u64(&mut self, tag: &[u8], v: u64) {
        self.absorb_bytes(tag, &v.to_le_bytes());
    }

    pub fn absorb_f64(&mut self, tag: &[u8], v: f64) {
        self.absorb_bytes(tag, &v.to_bits().to_le_bytes());
    }

    pub fn absorb_f64_slice(&mut self, tag: &[u8], xs: &[f64]) {
        self.absorb_tag(tag);
        self.h.update((xs.len() as u64).to_le_bytes());
        absorb_f64_bytes(&mut self.h, xs);
    }

    pub fn absorb_f64_2d(&mut self, tag: &[u8], rows: usize, cols: usize, xs: &[f64]) {
        self.absorb_tag(tag);
        self.h.update((rows as u64).to_le_bytes());
        self.h.update((cols as u64).to_le_bytes());
        absorb_f64_bytes(&mut self.h, xs);
    }

    pub fn finalize(self) -> Fingerprint {
        let out = self.h.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&out);
        Fingerprint(bytes)
    }

    // ------------------------------------------------------------------
    // Untagged write_* API — drop-in replacement for the formerly-separate
    // `StableHasher` (warm-start) and `CacheDigestBuilder` (latent_cache)
    // hashers. Callers that use this API are responsible for their own
    // type-disambiguation (typically by writing a leading namespace string
    // via `write_str`); the `absorb_*` family above prepends per-call tags
    // and is the safer choice for new code that does not need streaming
    // compatibility with hand-framed inputs.
    // ------------------------------------------------------------------

    pub fn write_bytes(&mut self, data: &[u8]) {
        self.h.update(data);
    }

    pub fn write_u8(&mut self, value: u8) {
        self.h.update([value]);
    }

    pub fn write_bool(&mut self, value: bool) {
        self.h.update([u8::from(value)]);
    }

    pub fn write_u64(&mut self, value: u64) {
        self.h.update(value.to_le_bytes());
    }

    pub fn write_usize(&mut self, value: usize) {
        self.h.update((value as u64).to_le_bytes());
    }

    pub fn write_f64(&mut self, value: f64) {
        // Normalize -0.0 to +0.0 so signed-zero comparison ambiguity does
        // not split cache buckets — matches the prior `StableHasher`
        // contract that warm-start keys depended on.
        let normalized = if value == 0.0 { 0.0 } else { value };
        self.h.update(normalized.to_bits().to_le_bytes());
    }

    pub fn write_str(&mut self, value: &str) {
        self.write_usize(value.len());
        self.h.update(value.as_bytes());
    }

    /// Absorb a length-prefixed `f64` slice using the per-element
    /// [`Fingerprinter::write_f64`] contract (so `-0.0` is normalized to
    /// `+0.0`). Canonical home for the byte-identical `len`-then-each-`f64`
    /// hashing that previously lived as module-local `hash_f64_slice` /
    /// `hash_vector` copies in `solver/latent_cache`. Use this — not
    /// [`Fingerprinter::absorb_f64_slice`], whose bulk path does **not**
    /// normalize signed zero — wherever warm-start keys depend on that
    /// normalization.
    pub fn write_f64_slice(&mut self, values: &[f64]) {
        self.write_usize(values.len());
        for &value in values {
            self.write_f64(value);
        }
    }

    /// Absorb a 1D `f64` array as `len` followed by every element via
    /// [`Fingerprinter::write_f64`]. Canonical home for the byte-identical
    /// `hash_vector` copy that previously lived in `solver/latent_cache`.
    pub fn write_f64_array1(&mut self, values: &ndarray::Array1<f64>) {
        self.write_usize(values.len());
        for &value in values.iter() {
            self.write_f64(value);
        }
    }

    /// Absorb a 2D `f64` array as `(nrows, ncols)` followed by every element in
    /// iteration order, each via [`Fingerprinter::write_f64`]. Canonical home
    /// for the byte-identical heuristic that previously lived as module-local
    /// `write_array2_fingerprint` (`solver/arrow_schur`) and `hash_matrix`
    /// (`solver/latent_cache`) copies.
    pub fn write_f64_array2(&mut self, values: &ndarray::Array2<f64>) {
        self.write_usize(values.nrows());
        self.write_usize(values.ncols());
        for &value in values.iter() {
            self.write_f64(value);
        }
    }

    /// Finalize and return the first 8 bytes of the SHA-256 digest as a
    /// little-endian `u64`. Used by callers that need a compact in-process
    /// identifier (manifold mode fingerprints, registry fingerprints, …)
    /// rather than the full 32-byte [`Fingerprint`].
    pub fn finish_u64(self) -> u64 {
        let out = self.h.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&out[..8]);
        u64::from_le_bytes(bytes)
    }

    /// Finalize and return a zero-padded 16-character hex representation
    /// of [`Fingerprinter::finish_u64`], suitable for embedding directly
    /// in cache-key strings.
    pub fn finish_hex(self) -> String {
        format!("{:016x}", self.finish_u64())
    }
}

/// Feed `xs` to the hasher in one bulk `update` instead of one 8-byte
/// `to_le_bytes` call per element. On little-endian hosts we reinterpret the
/// `&[f64]` storage directly as `&[u8]`; on big-endian hosts we fall back to
/// a per-element loop so the fingerprint stays endian-stable across machines.
#[inline]
fn absorb_f64_bytes(h: &mut Sha256, xs: &[f64]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: xs.as_ptr() is non-null/aligned (slice invariant); f64
        // has no padding and any bit pattern is a valid u8; size_of_val
        // covers exactly xs's bytes and the borrow lives within this call.
        let bytes = unsafe {
            std::slice::from_raw_parts(xs.as_ptr() as *const u8, std::mem::size_of_val(xs))
        };
        h.update(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &x in xs {
            h.update(x.to_bits().to_le_bytes());
        }
    }
}

impl Default for Fingerprinter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrips() {
        let mut fp = Fingerprinter::new();
        fp.absorb_str(b"family", "standard");
        fp.absorb_f64_slice(b"y", &[1.0, 2.0, 3.0]);
        let key = fp.finalize();
        let hex = key.to_hex();
        assert_eq!(hex.len(), 64);
        let parsed = Fingerprint::from_hex(&hex).unwrap();
        assert_eq!(key, parsed);
    }

    #[test]
    fn tagged_absorptions_dont_collide() {
        let mut a = Fingerprinter::new();
        a.absorb_f64(b"x", 0.5);
        let ka = a.finalize();

        let mut b = Fingerprinter::new();
        // Same bytes, different tag: must produce a different key.
        b.absorb_bytes(b"y", &0.5f64.to_bits().to_le_bytes());
        let kb = b.finalize();

        assert_ne!(ka, kb);
    }

    #[test]
    fn different_data_yields_different_keys() {
        let mut a = Fingerprinter::new();
        a.absorb_f64_slice(b"y", &[1.0, 2.0]);
        let mut b = Fingerprinter::new();
        b.absorb_f64_slice(b"y", &[1.0, 3.0]);
        assert_ne!(a.finalize(), b.finalize());
    }

    #[test]
    fn same_input_yields_same_key() {
        let mut a = Fingerprinter::new();
        a.absorb_str(b"f", "binomial");
        a.absorb_f64_2d(b"x", 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut b = Fingerprinter::new();
        b.absorb_str(b"f", "binomial");
        b.absorb_f64_2d(b"x", 2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(a.finalize(), b.finalize());
    }

    #[test]
    fn invalid_hex_rejected() {
        assert!(Fingerprint::from_hex("not hex").is_none());
        assert!(Fingerprint::from_hex(&"a".repeat(63)).is_none());
        assert!(Fingerprint::from_hex(&"z".repeat(64)).is_none());
    }
}
