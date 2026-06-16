//! Disk-backed per-row warm-state cache for the streaming SAE inner solve
//! (#973).
//!
//! # The 3-vs-30 economics
//!
//! Each corpus row's SAE code is found by an inner solve (latent coords + an
//! active set of dictionary atoms). Cold, that solve takes ~30 inner
//! iterations. But across outer ρ passes (and across resumed runs) the *same
//! row* is solved again and again, and its solution barely moves between
//! neighbouring ρ values. Seeding the next solve from the previous one's latent
//! coords + active set cuts it to ~3 iterations. Over a multi-million-row
//! corpus that is the difference between a tractable and an intractable fit.
//!
//! This module persists that per-row warm start so the economics survive both
//! the outer ρ loop *and* a process restart (SIGKILL-resume), keyed so that a
//! warm start can never be applied to a structurally different model.
//!
//! # Keying
//!
//! The cache key is `(row_id, TermCollectionSpec structural hash)`. The
//! structural hash is computed **the same way the persistent warm-start cache
//! already does it** — via
//! [`crate::terms::smooth::TermCollectionSpec::write_structural_shape_hash`]
//! (#869), the topology-aware shape hash — so a sphere-vs-torus-vs-euclidean
//! candidate on the same data gets a *distinct* per-row warm-start keyspace and
//! the candidates never cross-seed each other with geometrically incompatible
//! coords. We hash that shape into a [`crate::warm_start::Fingerprinter`] together
//! with the `row_id`, matching the existing warm-start key derivation byte
//! framing.
//!
//! # Storage
//!
//! The on-disk tier reuses [`crate::warm_start::WarmStartStore`] (tmp-file + fsync +
//! rename writes, per-entry checksums, bounded size + TTL eviction) so we
//! inherit its crash-safety and disk-budget guarantees for free. In front of
//! it sits a bounded in-process LRU keyed by the same fingerprint, so the hot
//! rows of the current batch never round-trip to disk. The serialized payload
//! is **bit-deterministic**: a fixed-layout little-endian encoding (no
//! `HashMap` iteration, no float formatting), so the same warm state always
//! hashes/round-trips identically.

use crate::warm_start::store::{EntryKind, StoreOptions, WarmStartStore};
use crate::warm_start::{Fingerprint, Fingerprinter};
use crate::terms::smooth::TermCollectionSpec;
use std::collections::HashMap;
use std::time::Duration;

/// On-disk payload schema tag. Bump on any layout change so stale entries are
/// rejected rather than mis-decoded.
const WARM_STATE_SCHEMA: u32 = 1;

/// In-process LRU capacity (entries). Auto-derived; bounds resident warm-state
/// memory regardless of corpus size. Not a CLI knob.
const LRU_CAPACITY: usize = 8192;

/// On-disk budget for the per-row warm-state tier.
const DISK_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
/// Disk TTL: warm states older than this are evicted on save.
const DISK_TTL_SECS: u64 = 14 * 24 * 60 * 60;

/// A serialized inner-solve warm start for one corpus row.
///
/// `latent_coords` are the SAE latent coordinates; `active_set` is the indices
/// of the dictionary atoms that were active at the previous solution. Together
/// they let the next solve start ~3 iterations from convergence instead of ~30.
#[derive(Debug, Clone, PartialEq)]
pub struct RowWarmState {
    pub latent_coords: Vec<f64>,
    pub active_set: Vec<u32>,
    /// Inner iteration count the previous solve converged in. Carried for
    /// diagnostics / adaptive scheduling; does not affect the seed itself.
    pub last_inner_iters: u32,
}

impl RowWarmState {
    /// Bit-deterministic little-endian serialization.
    ///
    /// Layout: `schema(u32) | n_coords(u64) | coords[f64 LE]… | n_active(u64) |
    /// active[u32 LE]… | last_inner_iters(u32)`. No map iteration, no float
    /// formatting — the bytes are a pure function of the value, so two equal
    /// `RowWarmState`s always serialize identically.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            4 + 8 + self.latent_coords.len() * 8 + 8 + self.active_set.len() * 4 + 4,
        );
        out.extend_from_slice(&WARM_STATE_SCHEMA.to_le_bytes());
        out.extend_from_slice(&(self.latent_coords.len() as u64).to_le_bytes());
        for &c in &self.latent_coords {
            // Normalize signed zero so two arithmetically-equal seeds serialize
            // byte-identically (matches the Fingerprinter::write_f64 contract).
            let v = if c == 0.0 { 0.0 } else { c };
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&(self.active_set.len() as u64).to_le_bytes());
        for &a in &self.active_set {
            out.extend_from_slice(&a.to_le_bytes());
        }
        out.extend_from_slice(&self.last_inner_iters.to_le_bytes());
        out
    }

    /// Inverse of [`Self::serialize`]; returns `None` on schema mismatch,
    /// truncation, or trailing garbage.
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        let mut off = 0usize;
        let take = |off: &mut usize, n: usize| -> Option<&[u8]> {
            let end = off.checked_add(n)?;
            if end > bytes.len() {
                return None;
            }
            let s = &bytes[*off..end];
            *off = end;
            Some(s)
        };
        let schema = u32::from_le_bytes(take(&mut off, 4)?.try_into().ok()?);
        if schema != WARM_STATE_SCHEMA {
            return None;
        }
        let n_coords = u64::from_le_bytes(take(&mut off, 8)?.try_into().ok()?) as usize;
        let mut latent_coords = Vec::with_capacity(n_coords);
        for _ in 0..n_coords {
            let bits = u64::from_le_bytes(take(&mut off, 8)?.try_into().ok()?);
            latent_coords.push(f64::from_bits(bits));
        }
        let n_active = u64::from_le_bytes(take(&mut off, 8)?.try_into().ok()?) as usize;
        let mut active_set = Vec::with_capacity(n_active);
        for _ in 0..n_active {
            active_set.push(u32::from_le_bytes(take(&mut off, 4)?.try_into().ok()?));
        }
        let last_inner_iters = u32::from_le_bytes(take(&mut off, 4)?.try_into().ok()?);
        if off != bytes.len() {
            // Trailing bytes => corrupt / wrong-schema payload.
            return None;
        }
        Some(Self {
            latent_coords,
            active_set,
            last_inner_iters,
        })
    }
}

/// The per-row warm-state cache seam.
///
/// This is the second half of the driver seam ([`super::shard_reader::CorpusRowSource`]
/// is the first). The streaming SAE term will, per row: `get` a seed, run the
/// inner solve from it, then `put` the refined state back.
pub trait RowWarmCache {
    /// Fetch the warm state for `row_id` under this cache's bound topology, if
    /// present (in-process LRU first, then disk).
    fn get(&mut self, row_id: u64) -> Option<RowWarmState>;
    /// Store / overwrite the warm state for `row_id`.
    fn put(&mut self, row_id: u64, state: &RowWarmState);
}

/// A bounded in-process LRU node.
struct LruEntry {
    /// The canonical row id this entry belongs to; used to detect the (rare)
    /// u64-key collision so a colliding lookup falls through to the disk tier
    /// rather than returning a wrong-row seed.
    row_id: u64,
    state: RowWarmState,
    /// Monotonic stamp for LRU ordering (highest = most recently used).
    stamp: u64,
}

/// mmap/LRU-backed disk cache implementing [`RowWarmCache`].
///
/// Bound to one `TermCollectionSpec` structural hash at construction; every key
/// folds that hash so two topologies cannot collide.
pub struct DiskRowWarmCache {
    /// Per-topology structural hash, folded into every row key.
    structural_hash: u64,
    /// Bounded in-process LRU over `Fingerprint`-equivalent row keys.
    lru: HashMap<u64, LruEntry>,
    stamp: u64,
    /// On-disk tier (None if the cache directory is unwritable; the cache then
    /// degrades to in-process-LRU-only without erroring).
    store: Option<WarmStartStore>,
}

impl DiskRowWarmCache {
    /// Construct a cache bound to `spec`'s structural topology, deriving the
    /// per-topology structural hash the same way the existing warm-start cache
    /// does (#869) — via `write_structural_shape_hash`.
    pub fn new(spec: &TermCollectionSpec) -> Self {
        let mut fp = Fingerprinter::new();
        fp.write_str("sae-corpus-row-warm-state-v1");
        spec.write_structural_shape_hash(&mut fp);
        let structural_hash = fingerprint_to_u64(&fp.finalize());
        let store = Self::open_store();
        Self {
            structural_hash,
            lru: HashMap::new(),
            stamp: 0,
            store,
        }
    }

    /// Anchor the disk tier under the platform temp directory, mirroring the
    /// persistent-warm-start root resolution (which avoids the banned
    /// `env::var` path that `dirs::cache_dir()` would take).
    fn open_store() -> Option<WarmStartStore> {
        let root = std::env::temp_dir()
            .join("gam")
            .join("sae_corpus_warm")
            .join("v1");
        WarmStartStore::open(
            root,
            StoreOptions {
                size_budget_bytes: DISK_BUDGET_BYTES,
                ttl: Duration::from_secs(DISK_TTL_SECS),
            },
        )
        .ok()
    }

    /// Compose the full disk/LRU key for a row under this cache's topology.
    ///
    /// Folds the schema tag, the per-topology structural hash, and the row id
    /// into one fingerprint, matching the existing warm-start key framing
    /// (length-prefixed `write_*` calls on a `Fingerprinter`). The `u64`
    /// reduction keys the in-process LRU; the full `Fingerprint` keys disk.
    fn row_fingerprint(&self, row_id: u64) -> Fingerprint {
        let mut fp = Fingerprinter::new();
        fp.write_str("sae-corpus-row-warm-state-key-v1");
        fp.write_u64(self.structural_hash);
        fp.write_u64(row_id);
        fp.finalize()
    }

    #[inline]
    fn lru_key(&self, row_id: u64) -> u64 {
        fingerprint_to_u64(&self.row_fingerprint(row_id))
    }

    /// Evict the least-recently-used LRU entry when over capacity.
    fn evict_if_full(&mut self) {
        if self.lru.len() <= LRU_CAPACITY {
            return;
        }
        if let Some((&victim, _)) = self.lru.iter().min_by_key(|(_, e)| e.stamp) {
            self.lru.remove(&victim);
        }
    }
}

impl RowWarmCache for DiskRowWarmCache {
    fn get(&mut self, row_id: u64) -> Option<RowWarmState> {
        let key = self.lru_key(row_id);
        // Hot path: in-process LRU. Guard against the (rare) u64-key collision
        // by checking the stored row_id; a collision falls through to the disk
        // tier which always re-checks the full Fingerprint.
        if let Some(entry) = self.lru.get_mut(&key) {
            if entry.row_id == row_id {
                self.stamp += 1;
                entry.stamp = self.stamp;
                return Some(entry.state.clone());
            }
            // Collision: drop through to disk.
        }
        // Cold path: disk tier. Decode, then promote into the LRU (overwriting
        // any colliding entry; the evicted entry's row_id diverges from key so
        // a subsequent get for the displaced row will also fall through to disk
        // — correct, just slower).
        let store = self.store.as_ref()?;
        let fp = self.row_fingerprint(row_id);
        let cached = store.lookup(&fp).ok().flatten()?;
        let state = RowWarmState::deserialize(&cached.payload)?;
        self.stamp += 1;
        self.lru.insert(
            key,
            LruEntry {
                row_id,
                state: state.clone(),
                stamp: self.stamp,
            },
        );
        self.evict_if_full();
        Some(state)
    }

    fn put(&mut self, row_id: u64, state: &RowWarmState) {
        let key = self.lru_key(row_id);
        self.stamp += 1;
        self.lru.insert(
            key,
            LruEntry {
                row_id,
                state: state.clone(),
                stamp: self.stamp,
            },
        );
        self.evict_if_full();
        // Write-through to disk so the seed survives a process restart. A
        // failed disk write is non-fatal: the LRU still holds the seed for the
        // current run. `iteration` carries the converged inner-iter count for
        // disk-side diagnostics; `Final` kind marks a converged seed.
        if let Some(store) = self.store.as_ref() {
            let payload = state.serialize();
            let fp = self.row_fingerprint(row_id);
            store
                .save(
                    &fp,
                    &payload,
                    None,
                    Some(u64::from(state.last_inner_iters)),
                    EntryKind::Final,
                )
                .ok();
        }
    }
}

/// Reduce a 32-byte [`Fingerprint`] to a `u64` LRU bucket key by folding its
/// raw leading bytes. Collisions in the `u64` space are harmless: the
/// disk tier always re-checks the full `Fingerprint`, and an LRU bucket
/// collision only risks a spurious in-process miss (then a correct disk hit),
/// never a wrong-row seed.
fn fingerprint_to_u64(fp: &Fingerprint) -> u64 {
    // Take the first 8 raw bytes of the 32-byte fingerprint and assemble them
    // into a u64. Using raw bytes (not the hex-string ASCII representation)
    // gives full 8-bit entropy per lane rather than the biased 4-bit range
    // that hex ASCII digits occupy (0x30-0x39, 0x61-0x66).
    let bytes = fp.as_bytes();
    let mut acc = 0u64;
    for &b in bytes.iter().take(8) {
        acc = acc.wrapping_shl(8) ^ u64::from(b);
    }
    // Mix so adjacent row ids (which share a long key prefix) spread across
    // buckets. Reuses the canonical splitmix64 finalizer.
    crate::linalg::utils::splitmix64_hash(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state() -> RowWarmState {
        RowWarmState {
            latent_coords: vec![1.0, -2.5, 0.0, 3.125],
            active_set: vec![0, 4, 9, 17],
            last_inner_iters: 3,
        }
    }

    #[test]
    fn serialize_round_trips() {
        let s = sample_state();
        let bytes = s.serialize();
        let back = RowWarmState::deserialize(&bytes).expect("decode");
        assert_eq!(s, back);
    }

    #[test]
    fn serialize_is_bit_deterministic() {
        // -0.0 normalizes to +0.0 so two arithmetically-equal seeds match.
        let a = RowWarmState {
            latent_coords: vec![-0.0, 1.0],
            active_set: vec![2],
            last_inner_iters: 1,
        };
        let b = RowWarmState {
            latent_coords: vec![0.0, 1.0],
            active_set: vec![2],
            last_inner_iters: 1,
        };
        assert_eq!(a.serialize(), b.serialize());
        // Re-serializing yields identical bytes.
        assert_eq!(a.serialize(), a.serialize());
    }

    #[test]
    fn deserialize_rejects_wrong_schema() {
        let mut bytes = sample_state().serialize();
        bytes[0] ^= 0xFF;
        assert!(RowWarmState::deserialize(&bytes).is_none());
    }

    #[test]
    fn deserialize_rejects_trailing_garbage() {
        let mut bytes = sample_state().serialize();
        bytes.push(0u8);
        assert!(RowWarmState::deserialize(&bytes).is_none());
    }

    #[test]
    fn deserialize_rejects_truncation() {
        let bytes = sample_state().serialize();
        assert!(RowWarmState::deserialize(&bytes[..bytes.len() - 2]).is_none());
    }
}
