//! A `Session` ties a `WarmStartStore` to a specific `Fingerprint` so callers
//! can resume + checkpoint a single fit without re-passing the key on every
//! call. One session corresponds to one in-flight fit; periodic checkpoints
//! overwrite a single run-id slot so we don't accumulate one entry per write.

use crate::cache::key::Fingerprint;
use crate::cache::store::{CachedEntry, EntryKind, WarmStartStore};
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
    pub entry: CachedEntry,
    pub source: LoadSource,
}

#[derive(Debug)]
pub struct Session {
    store: WarmStartStore,
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
    preloaded: Mutex<Option<CachedEntry>>,
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
            key,
            run_id,
            inner: Mutex::new(Inner {
                last_write: None,
                best_seen: None,
            }),
            preloaded: Mutex::new(None),
        }
    }

    /// Stash a near-match payload that the next [`Self::try_load`] call
    /// should return in preference to looking up this session's key.
    ///
    /// Used by the workflow dispatcher to seed a fresh fit's outer loop
    /// from a related but not-exact-fingerprint prior fit (e.g.,
    /// cross-validation folds of the same model). The exact-key keyspace
    /// remains untouched by this — checkpoint and finalize writes still
    /// go to the session's own key.
    pub fn preload(&self, entry: CachedEntry) {
        let mut slot = match self.preloaded.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        *slot = Some(entry);
    }

    pub fn key(&self) -> &Fingerprint {
        &self.key
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn store(&self) -> &WarmStartStore {
        &self.store
    }

    /// Read the best entry currently on disk for this session's key.
    /// Lookup is read-only against the store and may return entries from
    /// other runs (the whole point of cross-run resume).
    ///
    /// If a near-match seed has been preloaded via [`Self::preload`],
    /// the seed is returned in preference to the store lookup AND
    /// consumed (so subsequent calls fall back to the store). This
    /// makes the session a unified abstraction over "exact-key hit"
    /// and "hierarchical-prefix seed."
    pub fn try_load(&self) -> Option<CachedEntry> {
        self.try_load_with_source().map(|loaded| loaded.entry)
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
        self.store
            .lookup(&self.key)
            .ok()
            .flatten()
            .map(|entry| LoadedEntry {
                entry,
                source: LoadSource::Exact,
            })
    }

    /// Read the currently available warm-start entry without consuming a
    /// preloaded near-match seed.
    ///
    /// This is intentionally separate from [`Self::try_load`]: callers that
    /// only need to make a scheduling decision (for example, whether to run an
    /// expensive cold-start pilot) must not drain the preloaded seed that the
    /// outer optimizer is about to consume.
    pub fn peek_load(&self) -> Option<CachedEntry> {
        self.peek_load_with_source().map(|loaded| loaded.entry)
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
        self.store
            .lookup(&self.key)
            .ok()
            .flatten()
            .map(|entry| LoadedEntry {
                entry,
                source: LoadSource::Exact,
            })
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
        if !improves {
            if let Some(last) = guard.last_write {
                if now.duration_since(last) < MIN_CHECKPOINT_INTERVAL {
                    return false;
                }
            }
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
            Err(_) => false,
        }
    }

    /// Persist the end-of-fit result, promoting this session's slot to
    /// `EntryKind::Final`. Bypasses the rate limit.
    pub fn finalize(&self, payload: &[u8], objective: Option<f64>, iteration: Option<u64>) -> bool {
        self.store
            .save_overwrite(
                &self.key,
                &self.run_id,
                payload,
                objective,
                iteration,
                EntryKind::Final,
            )
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::key::Fingerprinter;
    use crate::cache::store::StoreOptions;

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

        let seeded = CachedEntry {
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

        let seeded = CachedEntry {
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

        // Second try_load: store fallback (seed already consumed).
        let second = s.try_load().expect("second call should fall back to store");
        assert_eq!(second.payload, b"exact");
    }

    #[test]
    fn peek_load_does_not_consume_preloaded_seed() {
        let (_d, s) = temp_session("preload-peek");
        let seeded = CachedEntry {
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
