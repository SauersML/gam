//! On-disk warm-start store.
//!
//! Persists periodic checkpoints of in-progress fits so a subsequent run
//! (possibly after a SIGKILL or weeks later) auto-resumes from the
//! best-known iterate. Keyed on a SHA-256 fingerprint of the data + fit
//! spec, so re-fitting the same model on the same data reuses the matching
//! persisted warm-start entry.
//!
//! The store does not choose its own root — [`WarmStartStore::open`] takes one.
//! High-level fitting accepts an explicit root and carries it through
//! [`ConfiguredWarmStartStore`]. There is no ambient temp/cache-directory
//! lookup and no implicit cross-process persistence: omitting the root keeps a
//! fit disk-silent.
//!
//! Layout under that root:
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
//! runs). `lookup` prefers a completed fit's [`EntryKind::Final`] write over
//! any [`EntryKind::Checkpoint`], takes the LATEST of several terminal writes
//! (which completed fit a resume carries is a provenance question, not a
//! quality one), and orders checkpoints by lowest objective.
//!
//! # The key identifies the problem; the entry identifies its producer
//!
//! The fingerprint is over `(data, spec)` and deliberately absorbs nothing
//! version-like, so a library upgrade that does not change the on-disk layout
//! keeps every user's warm start. That is right for a *seed* — a seed only has
//! to be close — but [`EntryKind::Final`] claims more than closeness: it claims
//! a converged optimization ended there, judged against the criterion the
//! writing code implements. Since the key cannot see that code, two builds
//! fitting the same model shared entries, and one build could resume the
//! other's terminus and ship it at zero outer iterations (#2625).
//!
//! Each entry therefore records the identity of the binary that wrote it, and
//! every read compares it: **a terminal certificate from a different build (or
//! from an unknown one) is returned as a [`EntryKind::Checkpoint`] instead.**
//! Reuse is untouched — the payload, objective and timestamps all survive, and
//! the iterate is still the best available seed. What is withdrawn is the right
//! to be shipped as a fit that this build's outer search never ran. The
//! comparison lives at the single point where metadata is deserialized, so
//! lookup, ranking and eviction cannot observe a foreign terminus and none of
//! them has to remember to ask.
//!
//! Consequently a warm hit may cost outer iterations it did not cost before,
//! and that is the intended price: SPEC-20 allows work to survive walls via
//! checkpoint/resume, and a resumed *seed* still has to converge here.
//!
//! Disk is bounded by [`StoreOptions::size_budget_bytes`] (default ~1 GiB);
//! oldest entries are evicted to fit. Entries older than
//! [`StoreOptions::ttl`] (default 30 days) are dropped on every save.

mod configured;
pub mod key;
pub mod session;
pub mod store;

pub use configured::ConfiguredWarmStartStore;
pub use key::{Fingerprint, Fingerprinter};
pub use session::Session;
pub use store::{EntryKind, StoreError, StoreOptions, WarmStartEntry, WarmStartStore};
