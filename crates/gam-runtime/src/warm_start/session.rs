//! A `Session` ties a `WarmStartStore` to a specific `Fingerprint` so callers
//! can resume + checkpoint a single fit without re-passing the key on every
//! call. One session corresponds to one in-flight fit; periodic checkpoints
//! overwrite a single run-id slot so we don't accumulate one entry per write.

use crate::warm_start::ConfiguredWarmStartStore;
use crate::warm_start::key::Fingerprint;
use crate::warm_start::store::{EntryKind, WarmStartEntry, WarmStartStore};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Minimum gap between checkpoint writes. Auto-derived; never less, so a
/// tight loop can't thrash disk. Improvements over the best-so-far always
/// bypass the rate limit — losing the best iterate to a hard crash is the
/// failure mode this whole module exists to prevent.
const MIN_CHECKPOINT_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadSource {
    Exact,
    Preloaded,
}

#[derive(Debug, Clone)]
pub struct LoadedEntry {
    pub entry: WarmStartEntry,
    pub source: LoadSource,
}

#[derive(Debug)]
pub struct Session {
    store: WarmStartStore,
    configured_store: Option<ConfiguredWarmStartStore>,
    key: Fingerprint,
    run_id: String,
    inner: Mutex<Inner>,
    /// Pre-loaded seed payload from a hierarchical near-match key.
    ///
    /// Populated by callers who looked up a related (but not exact-match)
    /// entry from a different key in the same store. The first call to
    /// [`Self::try_load`] returns and clears this slot — so the session
    /// can be used as a unified "load best seed, save under exact key"
    /// abstraction regardless of where the seed came from.
    preloaded: Mutex<Option<WarmStartEntry>>,
}

#[derive(Debug)]
struct Inner {
    last_write: Option<Instant>,
    best_seen: Option<f64>,
}

impl Session {
    pub fn open(store: WarmStartStore, key: Fingerprint) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let run_id = format!("ckpt-r{pid:x}-{nanos:x}");
        Self {
            store,
            configured_store: None,
            key,
            run_id,
            inner: Mutex::new(Inner {
                last_write: None,
                best_seen: None,
            }),
            preloaded: Mutex::new(None),
        }
    }

    pub(super) fn open_configured(
        store: WarmStartStore,
        key: Fingerprint,
        configured_store: ConfiguredWarmStartStore,
    ) -> Self {
        let mut session = Self::open(store, key);
        session.configured_store = Some(configured_store);
        session
    }

    fn configured_store_is_available(&self) -> bool {
        self.configured_store
            .as_ref()
            .is_none_or(ConfiguredWarmStartStore::is_available)
    }

    fn record_store_error(&self, operation: &str, error: &crate::warm_start::StoreError) {
        if let Some(configured_store) = &self.configured_store {
            configured_store.record_store_error(operation, error);
        }
    }

    pub fn key(&self) -> &Fingerprint {
        &self.key
    }

    /// Read the best available warm-start entry and report whether it came
    /// from this session's exact key or from a preloaded near-match seed.
    ///
    /// Callers that only need a seed can use [`Self::try_load`]. Callers that
    /// may skip expensive validation on a finalized exact hit need this source
    /// bit so a near-match prefix seed is never mistaken for a completed fit.
    pub fn try_load_with_source(&self) -> Option<LoadedEntry> {
        if let Ok(mut slot) = self.preloaded.lock()
            && let Some(entry) = slot.take()
        {
            return Some(LoadedEntry {
                entry,
                source: LoadSource::Preloaded,
            });
        }
        if !self.configured_store_is_available() {
            return None;
        }
        match self.store.lookup(&self.key) {
            Ok(Some(entry)) => Some(LoadedEntry {
                entry,
                source: LoadSource::Exact,
            }),
            Ok(None) => None,
            Err(error) => {
                self.record_store_error("load outer-iterate session", &error);
                None
            }
        }
    }

    /// Read the currently available warm-start entry with source metadata,
    /// without consuming a preloaded near-match seed.
    pub fn peek_load_with_source(&self) -> Option<LoadedEntry> {
        if let Ok(slot) = self.preloaded.lock()
            && let Some(entry) = slot.as_ref()
        {
            return Some(LoadedEntry {
                entry: entry.clone(),
                source: LoadSource::Preloaded,
            });
        }
        if !self.configured_store_is_available() {
            return None;
        }
        match self.store.lookup(&self.key) {
            Ok(Some(entry)) => Some(LoadedEntry {
                entry,
                source: LoadSource::Exact,
            }),
            Ok(None) => None,
            Err(error) => {
                self.record_store_error("peek outer-iterate session", &error);
                None
            }
        }
    }

    /// Persist a mid-fit checkpoint. Rate-limited; returns true if a write
    /// actually happened. Always writes when the new objective strictly
    /// improves on the best-so-far observed in this session.
    pub fn checkpoint(
        &self,
        payload: &[u8],
        objective: Option<f64>,
        iteration: Option<u64>,
    ) -> bool {
        if !self.configured_store_is_available() {
            return false;
        }
        let now = Instant::now();
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let improves = match (objective, guard.best_seen) {
            (Some(o), Some(b)) => o < b - 1e-12,
            (Some(_), None) => true,
            _ => false,
        };
        if !improves
            && let Some(last) = guard.last_write
            && now.duration_since(last) < MIN_CHECKPOINT_INTERVAL
        {
            return false;
        }
        match self.store.save_overwrite(
            &self.key,
            &self.run_id,
            payload,
            objective,
            iteration,
            EntryKind::Checkpoint,
        ) {
            Ok(()) => {
                guard.last_write = Some(now);
                if let Some(o) = objective {
                    guard.best_seen = Some(match guard.best_seen {
                        Some(b) => b.min(o),
                        None => o,
                    });
                }
                true
            }
            Err(error) => {
                self.record_store_error("checkpoint outer-iterate session", &error);
                false
            }
        }
    }

    /// Persist the end-of-fit result, promoting this session's slot to
    /// `EntryKind::Final`. Bypasses the rate limit.
    pub fn finalize(&self, payload: &[u8], objective: Option<f64>, iteration: Option<u64>) -> bool {
        if !self.configured_store_is_available() {
            return false;
        }
        match self.store.save_overwrite(
            &self.key,
            &self.run_id,
            payload,
            objective,
            iteration,
            EntryKind::Final,
        ) {
            Ok(()) => true,
            Err(error) => {
                self.record_store_error("finalize outer-iterate session", &error);
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::warm_start::key::Fingerprinter;
    use crate::warm_start::store::StoreOptions;

    fn temp_session(label: &str) -> (tempfile::TempDir, Session) {
        let dir = tempfile::tempdir().unwrap();
        let store = WarmStartStore::open(
            dir.path().to_path_buf(),
            StoreOptions {
                size_budget_bytes: 1024 * 1024,
                ttl: Duration::from_secs(60),
            },
        )
        .unwrap();
        let mut fp = Fingerprinter::new();
        fp.absorb_str(b"label", label);
        let key = fp.finalize();
        let s = Session::open(store, key);
        (dir, s)
    }

    #[test]
    fn checkpoint_then_load() {
        let (_d, s) = temp_session("ckpt");
        assert!(s.checkpoint(b"iter-1", Some(2.0), Some(1)));
        let got = s.try_load().unwrap();
        assert_eq!(got.payload, b"iter-1");
        assert_eq!(got.objective, Some(2.0));
        assert_eq!(got.kind, EntryKind::Checkpoint);
    }

    #[test]
    fn improving_objective_bypasses_rate_limit() {
        let (_d, s) = temp_session("improve");
        assert!(s.checkpoint(b"a", Some(5.0), Some(1)));
        // Immediately better objective — must write even though rate-limit
        // window is open.
        assert!(s.checkpoint(b"b", Some(3.0), Some(2)));
        let got = s.try_load().unwrap();
        assert_eq!(got.payload, b"b");
        assert_eq!(got.objective, Some(3.0));
    }

    #[test]
    fn non_improving_writes_are_throttled() {
        let (_d, s) = temp_session("throttle");
        assert!(s.checkpoint(b"a", Some(2.0), Some(1)));
        // Worse objective inside the rate window — should be suppressed.
        assert!(!s.checkpoint(b"b", Some(5.0), Some(2)));
        // Disk still shows the better iterate.
        let got = s.try_load().unwrap();
        assert_eq!(got.payload, b"a");
    }

    #[test]
    fn finalize_promotes_to_final_kind() {
        let (_d, s) = temp_session("final");
        s.checkpoint(b"ckpt", Some(2.0), Some(1));
        s.finalize(b"done", Some(1.0), Some(5));
        let got = s.try_load().unwrap();
        assert_eq!(got.payload, b"done");
        assert_eq!(got.kind, EntryKind::Final);
    }

    #[test]
    fn preload_takes_precedence_over_store_lookup() {
        // Hierarchical near-match semantics: when a session is opened on
        // a fresh key (no entry) but preloaded with a near-match payload
        // from a different key, try_load returns the preloaded entry.
        let (_d, s) = temp_session("preload-empty");
        assert!(s.try_load().is_none(), "fresh key should have no entry");

        let seeded = WarmStartEntry {
            payload: b"from-prefix".to_vec(),
            objective: Some(7.0),
            iteration: Some(42),
            kind: EntryKind::Final,
            written_unix_secs: 0,
        };
        s.preload(seeded);

        let got = s.try_load().expect("preloaded seed should be returned");
        assert_eq!(got.payload, b"from-prefix");
        assert_eq!(got.objective, Some(7.0));
    }

    #[test]
    fn preload_consumed_on_first_try_load() {
        // The preload slot is consumed after one read so subsequent calls
        // fall back to the store. This makes the session a unified
        // "load best seed, save under exact key" abstraction without
        // duplicating reads.
        let (_d, s) = temp_session("preload-consume");
        s.checkpoint(b"exact", Some(2.0), Some(5));

        let seeded = WarmStartEntry {
            payload: b"seed".to_vec(),
            objective: Some(99.0),
            iteration: Some(1),
            kind: EntryKind::Checkpoint,
            written_unix_secs: 0,
        };
        s.preload(seeded);

        // First try_load: seed (preferred over store).
        let first = s.try_load().expect("first call should return seed");
        assert_eq!(first.payload, b"seed");

        // Second try_load: store lookup after the seed is consumed.
        let second = s.try_load().expect("second call should read from store");
        assert_eq!(second.payload, b"exact");
    }

    #[test]
    fn peek_load_does_not_consume_preloaded_seed() {
        let (_d, s) = temp_session("preload-peek");
        let seeded = WarmStartEntry {
            payload: b"seed".to_vec(),
            objective: Some(3.0),
            iteration: Some(9),
            kind: EntryKind::Final,
            written_unix_secs: 0,
        };
        s.preload(seeded);

        let peeked = s
            .peek_load_with_source()
            .expect("peek should see preloaded seed");
        assert_eq!(peeked.entry.payload, b"seed");
        assert_eq!(peeked.source, LoadSource::Preloaded);

        let loaded = s
            .try_load()
            .expect("try_load should still receive the preloaded seed");
        assert_eq!(loaded.payload, b"seed");
        assert!(
            s.try_load().is_none(),
            "preloaded seed should be consumed only by try_load"
        );
    }

    #[test]
    fn second_session_reads_first_session_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let mut fp = Fingerprinter::new();
        fp.absorb_str(b"k", "shared");
        let key = fp.finalize();

        let store_a = WarmStartStore::open(
            dir.path().to_path_buf(),
            StoreOptions {
                size_budget_bytes: 1024 * 1024,
                ttl: Duration::from_secs(60),
            },
        )
        .unwrap();
        let s_a = Session::open(store_a, key);
        s_a.checkpoint(b"from-a", Some(1.0), Some(3));

        // Simulate a fresh process starting later.
        let store_b = WarmStartStore::open(
            dir.path().to_path_buf(),
            StoreOptions {
                size_budget_bytes: 1024 * 1024,
                ttl: Duration::from_secs(60),
            },
        )
        .unwrap();
        let s_b = Session::open(store_b, key);
        let got = s_b.try_load().unwrap();
        assert_eq!(got.payload, b"from-a");
        assert_eq!(got.objective, Some(1.0));
    }
}
