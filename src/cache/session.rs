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

pub struct Session {
    store: WarmStartStore,
    key: Fingerprint,
    run_id: String,
    inner: Mutex<Inner>,
}

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
        }
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
    /// Lookup is read-only and may return entries from other runs (the
    /// whole point of cross-run resume).
    pub fn try_load(&self) -> Option<CachedEntry> {
        self.store.lookup(&self.key).ok().flatten()
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
    pub fn finalize(
        &self,
        payload: &[u8],
        objective: Option<f64>,
        iteration: Option<u64>,
    ) -> bool {
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
