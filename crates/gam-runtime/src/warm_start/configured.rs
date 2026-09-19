//! Explicit, lazily opened warm-start storage.
//!
//! A configured store never discovers a root from process-global state. The
//! caller supplies the exact path, and clones share the one open decision and
//! the resulting [`WarmStartStore`]. Opening is lazy so parsing or validating a
//! fit request does not create directories.

use super::{Fingerprint, Session, StoreError, StoreOptions, WarmStartStore};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

/// A best-effort warm-start store rooted at an explicit caller-owned path.
///
/// Persistence is an optimization, so an unavailable filesystem never makes a
/// fit fail. The first attempted use opens the exact configured root and records
/// either the store or its permanent absence for this fit. Every clone observes
/// that same decision, preventing repeated directory scans and repeated
/// diagnostics.
#[derive(Clone, Debug)]
pub struct ConfiguredWarmStartStore {
    root: PathBuf,
    options: StoreOptions,
    opened: Arc<OnceLock<Option<WarmStartStore>>>,
    available: Arc<AtomicBool>,
}

impl ConfiguredWarmStartStore {
    /// Configure a store without touching the filesystem.
    pub fn new(root: PathBuf, options: StoreOptions) -> Self {
        Self {
            root,
            options,
            opened: Arc::new(OnceLock::new()),
            available: Arc::new(AtomicBool::new(true)),
        }
    }

    /// The exact root supplied by the caller. No canonicalization, joining, or
    /// environment-dependent relocation is performed.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Return the shared opened store, or `None` when persistence is
    /// unavailable.
    ///
    /// This is the sole owner of the open-failure decision and diagnostic.
    /// Failures are memoized: a fit proceeds cold and never repeatedly probes a
    /// root that already refused the configured store.
    pub fn store(&self) -> Option<&WarmStartStore> {
        if !self.available.load(Ordering::Relaxed) {
            return None;
        }
        self.opened
            .get_or_init(
                || match WarmStartStore::open(self.root.clone(), self.options.clone()) {
                    Ok(store) => {
                        log::debug!(
                            "[warm-start-cache] opened explicit root={}",
                            self.root.display()
                        );
                        Some(store)
                    }
                    Err(error) => {
                        self.mark_unavailable("open", &error);
                        None
                    }
                },
            )
            .as_ref()
    }

    /// Open a keyed session governed by this capability's shared availability
    /// decision.
    ///
    /// Session reads and writes report filesystem refusal back to this owner,
    /// so one failed outer-iterate checkpoint disables every record, artifact,
    /// and session operation belonging to the fit.
    pub fn open_session(&self, key: Fingerprint) -> Option<Arc<Session>> {
        let store = self.store()?.clone();
        Some(Arc::new(Session::open_configured(store, key, self.clone())))
    }

    pub(crate) fn is_available(&self) -> bool {
        self.available.load(Ordering::Relaxed)
    }

    pub(crate) fn record_store_error(&self, operation: &str, error: &StoreError) {
        match error {
            StoreError::Io(error) => self.mark_unavailable(operation, error),
            StoreError::Json(error) => {
                log::debug!(
                    "[warm-start-cache] persistence defect operation={} explicit_root={}: {}",
                    operation,
                    self.root.display(),
                    error
                );
            }
        }
    }

    /// Permanently disable this fit's configured persistence after the
    /// filesystem refuses an operation.
    ///
    /// The first refusal owns the sole diagnostic; subsequent loads/stores and
    /// every clone become cold no-ops. Callers should use this only for
    /// environmental I/O failures, not serialization or contract defects.
    pub fn mark_unavailable(&self, operation: &str, error: &dyn std::fmt::Display) {
        if self.available.swap(false, Ordering::Relaxed) {
            log::debug!(
                "[warm-start-cache] persistence unavailable operation={} explicit_root={}: {}; \
                 continuing without on-disk warm starts",
                operation,
                self.root.display(),
                error
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn options() -> StoreOptions {
        StoreOptions {
            size_budget_bytes: 1024 * 1024,
            ttl: Duration::from_secs(60),
        }
    }

    #[test]
    fn explicit_configuration_is_lazy_and_uses_the_exact_root() {
        let parent = tempfile::tempdir().expect("create isolated parent");
        let root = parent.path().join("caller-chosen").join("warm");
        let configured = ConfiguredWarmStartStore::new(root.clone(), options());

        assert_eq!(configured.root(), root);
        assert!(
            !root.exists(),
            "configuration parsing must not create the store root"
        );

        let opened = configured.store().expect("explicit root must open");
        assert_eq!(opened.root(), root);
        assert!(root.is_dir());
        let cloned = configured.clone();
        assert!(
            std::ptr::eq(opened, cloned.store().expect("clone shares opened store")),
            "every clone must reuse one opened store handle"
        );
    }

    #[test]
    fn unavailable_root_is_one_memoized_best_effort_decision() {
        let parent = tempfile::tempdir().expect("create isolated parent");
        let blocking_file = parent.path().join("not-a-directory");
        std::fs::write(&blocking_file, b"x").expect("create blocking file");
        let root = blocking_file.join("warm");
        let configured = ConfiguredWarmStartStore::new(root, options());

        assert!(configured.store().is_none());

        // Make the path usable after the first refusal. A second call must not
        // retry or mint a different per-clone decision.
        std::fs::remove_file(&blocking_file).expect("remove blocking file");
        std::fs::create_dir(&blocking_file).expect("replace it with a directory");
        assert!(configured.clone().store().is_none());
    }

    #[test]
    fn session_write_refusal_disables_the_shared_capability() {
        let parent = tempfile::tempdir().expect("create isolated parent");
        let root = parent.path().join("warm");
        let configured = ConfiguredWarmStartStore::new(root.clone(), options());
        let mut fingerprinter = crate::warm_start::Fingerprinter::new();
        fingerprinter.absorb_str(b"test", "configured-session");
        let session = configured
            .open_session(fingerprinter.finalize())
            .expect("open configured session");

        std::fs::remove_dir(&root).expect("remove empty opened root");
        std::fs::write(&root, b"not a directory").expect("block the configured root");
        assert!(!session.finalize(b"payload", None, None));
        assert!(
            configured.store().is_none(),
            "the session must report filesystem refusal to the shared owner"
        );

        std::fs::remove_file(&root).expect("remove blocking file");
        std::fs::create_dir(&root).expect("make the root usable again");
        assert!(
            !session.finalize(b"payload", None, None),
            "an unavailable capability must not retry through an existing session"
        );
    }
}
