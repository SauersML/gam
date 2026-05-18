//! On-disk warm-start cache.
//!
//! Persists periodic checkpoints of in-progress fits so a subsequent run
//! (possibly after a SIGKILL or weeks later) auto-resumes from the
//! best-known iterate. Keyed on a SHA-256 fingerprint of the data + fit
//! spec, so re-fitting the same model on the same data reuses the matching
//! persisted warm-start entry.
//!
//! Layout under `dirs::cache_dir()/gam/warm/v1/`:
//!
//! ```text
//! <keyhex>/
//!   <runid>.json    metadata (objective, iter, checksum, kind)
//!   <runid>.bin     opaque payload bytes
//! ```
//!
//! All writes are tmp-file + fsync + rename, so a hard crash leaves either
//! the pre-write state or a fully-written entry on disk — never half-written.
//! Per-entry SHA-256 checksums catch any residual corruption.
//!
//! Multiple entries can coexist for one key (concurrent fits, prior aborted
//! runs). `lookup` picks the lowest-objective entry; ties prefer
//! [`EntryKind::Final`] over [`EntryKind::Checkpoint`], then latest mtime.
//!
//! Disk is bounded by [`StoreOptions::size_budget_bytes`] (default ~1 GiB);
//! oldest entries are evicted to fit. Entries older than
//! [`StoreOptions::ttl`] (default 30 days) are dropped on every save.

pub mod key;
pub mod session;
pub mod store;

pub use key::{Fingerprint, Fingerprinter};
pub use session::Session;
pub use store::{CachedEntry, EntryKind, StoreError, StoreOptions, WarmStartStore};
