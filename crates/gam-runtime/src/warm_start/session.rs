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

    /// Stash a near-match payload that the next [`Self::try_load`] call
    /// should return in preference to looking up this session's key.
    ///
    /// Used by the workflow dispatcher to seed a fresh fit's outer loop
    /// from a related but not-exact-fingerprint prior fit (e.g.,
    /// cross-validation folds of the same model). The exact-key keyspace
    /// remains untouched by this — checkpoint and finalize writes still
    /// go to the session's own key.
    pub fn preload(&self, entry: WarmStartEntry) {
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

    /// Read the best entry currently on disk for this session's key.
    /// Lookup is read-only against the store and may return entries from
    /// other runs (the whole point of cross-run resume).
    ///
    /// If a near-match seed has been preloaded via [`Self::preload`],
    /// the seed is returned in preference to the store lookup AND
    /// consumed (so subsequent calls fall back to the store). This
    /// makes the session a unified abstraction over "exact-key hit"
    /// and "hierarchical-prefix seed."
    pub fn try_load(&self) -> Option<WarmStartEntry> {
        self.try_load_with_source().map(|loaded| loaded.entry)
    }

    /// Read the best available warm-start entry and report whether it came
    /// from this session's exact key or from a preloaded near-match seed.
    ///
    /// Callers that only need a seed can use `Self::try_load`. Callers that
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

