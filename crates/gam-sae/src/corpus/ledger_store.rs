//! Run-level persistence for the structure-evidence ledger (#993 item 3:
//! ledger ownership in the corpus loop).
//!
//! # Who owns the ledger
//!
//! The division of labor, settled here:
//!
//! * **Within a run**, the structure-search engine
//!   (`gam_solve::structure_search`) owns the ledger mutably — it
//!   registers claims and absorbs gate evidence as moves are adjudicated.
//! * **Across runs and shards**, THIS store owns it: the ledger is loaded
//!   before the shard loop, handed to the engine, and saved back after
//!   every shard — the same SIGKILL-resume contract the per-row warm
//!   state ([`super::warm_state`]) already honors, through the same
//!   crash-safe [`WarmStartStore`] tier (tmp-file + fsync + rename,
//!   per-entry checksums). Anytime validity is exactly what makes this
//!   sound: an e-process resumed mid-stream is still an e-process, so a
//!   killed-and-restarted discovery run loses compute, never validity.
//! * **At the chosen stop**, the caller certifies:
//!   `sae::identifiability::dictionary_report(model, &ledger, α)` pairs
//!   the e-BH structure certificate with the residual-gauge report. The
//!   default level is [`STRUCTURE_CERTIFICATE_ALPHA`].
//!
//! # Keying
//!
//! Keyed by the `TermCollectionSpec` structural shape hash, exactly like
//! the per-row warm state (#869 derivation): evidence accumulated for one
//! topology can never be replayed against a structurally different model
//! — that would void the e-process guarantee (the claims would not even
//! refer to the same atoms). A torus candidate and a sphere candidate on
//! the same corpus carry independent ledgers.
//!
//! # Corruption is loud
//!
//! A missing ledger is a fresh start ([`StructureLedger::new`]); a ledger
//! that EXISTS but fails to decode is an error, never a silent reset —
//! silently discarding accumulated evidence would let a crash launder a
//! claim's history and restart its e-process from 1, which is exactly the
//! optional-stopping abuse the ledger exists to prevent.

use crate::inference::structure_evidence::StructureLedger;
use gam_terms::smooth::TermCollectionSpec;
use gam_runtime::warm_start::store::{EntryKind, StoreOptions, WarmStartStore};
use gam_runtime::warm_start::{Fingerprint, Fingerprinter};
use std::time::Duration;

/// The default level for the dictionary's e-BH structure certificate.
/// 0.05 is the conventional reporting level for the certificate's FDR;
/// the e-process machinery is level-agnostic (the ledger stores log
/// e-values, and a consumer may certify at any α post hoc — that re-read
/// is safe BECAUSE the entries are e-values).
pub const STRUCTURE_CERTIFICATE_ALPHA: f64 = 0.05;

/// On-disk budget for ledgers. Ledgers are tiny (a few floats per claim);
/// 16 MiB holds thousands of candidate topologies' ledgers.
const LEDGER_DISK_BUDGET_BYTES: u64 = 16 * 1024 * 1024;
/// Ledger TTL matches the warm-state tier: stale discovery runs age out
/// together with their seeds.
const LEDGER_DISK_TTL_SECS: u64 = 14 * 24 * 60 * 60;
/// Fixed run-id: each topology has ONE canonical ledger entry, updated in
/// place atomically (`save_overwrite`), never a pile of generations.
const LEDGER_RUN_ID: &str = "structure-ledger";

/// Persistent, topology-keyed store for one dictionary's
/// [`StructureLedger`].
pub struct LedgerStore {
    key: Fingerprint,
    /// `None` when the cache directory is unwritable; the store then
    /// degrades to in-memory-only (load = fresh, save = no-op) without
    /// erroring — matching the warm-state tier's degradation.
    store: Option<WarmStartStore>,
}

impl LedgerStore {
    /// Bind a store to `spec`'s structural topology (the #869 shape-hash
    /// derivation, same as the per-row warm state).
    pub fn new(spec: &TermCollectionSpec) -> Self {
        let mut fp = Fingerprinter::new();
        fp.write_str("sae-structure-ledger-key-v1");
        spec.write_structural_shape_hash(&mut fp);
        Self::from_fingerprint(fp.finalize())
    }

    /// Bind by a precomputed structural hash where `structural_hash` is the
    /// raw value produced by hashing only [`TermCollectionSpec::write_structural_shape_hash`]
    /// into a [`Fingerprinter`] and reducing to `u64` — NOT the `structural_hash`
    /// field of [`super::warm_state::DiskRowWarmCache`], which folds an additional
    /// namespace prefix (`"sae-corpus-row-warm-state-v1"`) before calling
    /// `write_structural_shape_hash` and therefore produces a different value.
    /// Passing the warm-cache's field here will silently bind a different key
    /// than [`Self::new`] and cause a ledger miss / fresh-start on every load.
    /// Prefer [`Self::new`] whenever a [`TermCollectionSpec`] is in scope.
    pub fn from_structural_hash(structural_hash: u64) -> Self {
        let mut fp = Fingerprinter::new();
        fp.write_str("sae-structure-ledger-key-v1");
        fp.write_u64(structural_hash);
        Self::from_fingerprint(fp.finalize())
    }

    fn from_fingerprint(key: Fingerprint) -> Self {
        let root = std::env::temp_dir()
            .join("gam")
            .join("sae_structure_ledger")
            .join("v1");
        let store = WarmStartStore::open(
            root,
            StoreOptions {
                size_budget_bytes: LEDGER_DISK_BUDGET_BYTES,
                ttl: Duration::from_secs(LEDGER_DISK_TTL_SECS),
            },
        )
        .ok();
        Self { key, store }
    }

    /// Load this topology's ledger. Absent (or unwritable tier) ⇒ a fresh
    /// empty ledger; PRESENT-BUT-UNDECODABLE ⇒ `Err` (see module docs:
    /// silent evidence loss is the one forbidden failure mode).
    pub fn load(&self) -> Result<StructureLedger, String> {
        let Some(store) = self.store.as_ref() else {
            return Ok(StructureLedger::new());
        };
        match store.lookup_latest(&self.key) {
            Ok(Some(entry)) => deserialize_ledger(&entry.payload),
            Ok(None) => Ok(StructureLedger::new()),
            Err(e) => Err(format!("LedgerStore::load: store lookup failed: {e:?}")),
        }
    }

    /// Persist the ledger (atomic in-place overwrite of the canonical
    /// entry). Call after every absorbed shard — the write is small and
    /// the crash-safety contract is per-write. `iteration` is stamped
    /// with the claim count for disk-side diagnostics.
    pub fn save(&self, ledger: &StructureLedger) -> Result<(), String> {
        let Some(store) = self.store.as_ref() else {
            return Ok(());
        };
        let payload = serialize_ledger(ledger)?;
        store
            .save_overwrite(
                &self.key,
                LEDGER_RUN_ID,
                &payload,
                None,
                Some(ledger.claims().len() as u64),
                EntryKind::Final,
            )
            .map_err(|e| format!("LedgerStore::save: store write failed: {e:?}"))
    }
}

/// Canonical ledger payload encoding (JSON via the ledger's serde derive;
/// the store layer adds checksums, so decode failures here mean schema
/// drift, not bit rot).
pub fn serialize_ledger(ledger: &StructureLedger) -> Result<Vec<u8>, String> {
    serde_json::to_vec(ledger).map_err(|e| format!("ledger serialization failed: {e}"))
}

/// Inverse of [`serialize_ledger`].
pub fn deserialize_ledger(bytes: &[u8]) -> Result<StructureLedger, String> {
    serde_json::from_slice(bytes).map_err(|e| {
        format!(
            "ledger payload exists but failed to decode ({e}); refusing to silently reset \
             accumulated evidence — delete the entry explicitly if a fresh start is intended"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::structure_evidence::ClaimKind;

    /// The canonical payload round-trips with evidence intact — the
    /// resume contract at the encoding layer (the disk tier underneath is
    /// the already-tested WarmStartStore).
    #[test]
    fn ledger_payload_round_trips_with_evidence() {
        let mut ledger = StructureLedger::new();
        let idx = ledger.register(ClaimKind::AtomExists { atom: 7 });
        ledger.absorb_log(idx, 1.5).unwrap();
        ledger.absorb_log(idx, 0.25).unwrap();
        let edge = ledger.register(ClaimKind::BindingEdge { a: 1, b: 3 });
        ledger.absorb_log(edge, -0.4).unwrap();

        let bytes = serialize_ledger(&ledger).expect("encode");
        let back = deserialize_ledger(&bytes).expect("decode");
        assert_eq!(back.claims().len(), 2);
        assert_eq!(back.claims()[idx].kind, ClaimKind::AtomExists { atom: 7 });
        assert_eq!(back.claims()[idx].evidence.steps(), 2);
        assert!((back.claims()[idx].evidence.log_evidence() - 1.75).abs() < 1e-12);
        assert!((back.claims()[edge].evidence.log_evidence() + 0.4).abs() < 1e-12);
    }

    /// Garbage payloads are a loud error, never a silent fresh ledger.
    #[test]
    fn corrupt_payload_is_an_error_not_a_reset() {
        let err = deserialize_ledger(b"{not json").expect_err("must refuse");
        assert!(err.contains("refusing to silently reset"));
    }

    /// Different structural hashes bind different keys (no cross-topology
    /// evidence replay).
    #[test]
    fn distinct_topologies_get_distinct_keys() {
        let a = LedgerStore::from_structural_hash(0xDEAD_BEEF);
        let b = LedgerStore::from_structural_hash(0xFEED_FACE);
        assert_ne!(a.key, b.key);
    }
}
