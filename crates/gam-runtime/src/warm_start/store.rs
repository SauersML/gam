//! Filesystem store for warm-start entries.
//!
//! Each entry is a `(<runid>.json, <runid>.bin)` pair inside a per-key
//! directory. Writes go through a temp-file → fsync → rename sequence so a
//! crash mid-write never leaves a half-written entry visible to readers.
//! Per-entry SHA-256 checksums catch any residual corruption.
//!
//! See [`crate::warm_start`] for the public API summary.

use crate::warm_start::key::Fingerprint;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// On-disk schema version. Bump on incompatible format changes; old entries
/// are then ignored at read time and evicted on the next save.
pub(crate) const SCHEMA_VERSION: u32 = 1;

/// Default disk-budget for the whole warm-start store root (~1 GiB).
pub(crate) const DEFAULT_SIZE_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;

/// Default TTL — entries untouched for this long are dropped.
pub(crate) const DEFAULT_TTL_SECS: u64 = 60 * 60 * 24 * 30;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Entry returned from [`WarmStartStore::lookup`].
#[derive(Debug, Clone)]
pub struct WarmStartEntry {
    pub payload: Vec<u8>,
    pub objective: Option<f64>,
    pub iteration: Option<u64>,
    pub written_unix_secs: u64,
    pub kind: EntryKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryKind {
    /// Mid-fit checkpoint — fit was alive when written.
    Checkpoint,
    /// End-of-fit — fit terminated successfully.
    Final,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OnDiskMeta {
    schema_version: u32,
    written_unix_secs: u64,
    /// Nanosecond component of the write timestamp. Used to break ties in
    /// LRU eviction so entries written within the same second don't sort
    /// arbitrarily.
    #[serde(default)]
    written_nanos: u32,
    objective: Option<f64>,
    iteration: Option<u64>,
    kind: EntryKind,
    checksum_hex: String,
    payload_bytes: u64,
    /// Set when a lookup has reused this entry. Eviction keeps recently reused
    /// entries behind never-hit writes when a tight budget forces a choice.
    #[serde(default)]
    accessed: bool,
    /// Last-access timestamp (unix seconds + nanos). Distinct from the
    /// immutable `written_*` creation stamp: a lookup that reuses this entry
    /// bumps the access stamp (refreshing its TTL so hot entries survive)
    /// WITHOUT touching `written_*`. Keeping the two separate is required for
    /// correctness — `lookup_latest`/`entry_newer` order by the immutable
    /// creation stamp, so if a read moved `written_*` forward the merely
    /// *read* entry would masquerade as the most-recently-*written* one. Zero
    /// (the serde default for entries written before this field existed, and
    /// for never-reused entries) means "no access newer than creation": TTL
    /// then falls back to `written_*`.
    #[serde(default)]
    accessed_unix_secs: u64,
    #[serde(default)]
    accessed_nanos: u32,
}

/// Effective activity timestamp (nanoseconds since the unix epoch): the more
/// recent of the immutable creation stamp and the last-access stamp. TTL
/// expiry is measured from this so a reused entry stays alive, while ordering
/// (`entry_newer`) keys on `written_*` alone.
fn meta_activity_nanos(meta: &OnDiskMeta) -> u128 {
    let written = (meta.written_unix_secs as u128) * 1_000_000_000u128 + meta.written_nanos as u128;
    let accessed =
        (meta.accessed_unix_secs as u128) * 1_000_000_000u128 + meta.accessed_nanos as u128;
    written.max(accessed)
}

#[derive(Debug, Clone)]
pub struct StoreOptions {
    pub size_budget_bytes: u64,
    pub ttl: Duration,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            size_budget_bytes: DEFAULT_SIZE_BUDGET_BYTES,
            ttl: Duration::from_secs(DEFAULT_TTL_SECS),
        }
    }
}

#[derive(Debug)]
pub struct WarmStartStore {
    root: PathBuf,
    opts: StoreOptions,
    /// Per-store metadata index. It is populated lazily and shared by clones so
    /// checkpoint-heavy sessions do not repeatedly open every metadata JSON.
    index: Arc<Mutex<MetadataIndex>>,
    /// Approximate sum of bytes written under `root`. Used to throttle the
    /// full directory-scanning eviction in [`Self::save_overwrite`] — see
    /// `EVICT_EVERY_N_SAVES`. The counter resyncs to ground truth after every
    /// triggered sweep. Shared across clones (`Arc`) so the eviction throttle
    /// survives the per-operation store reuse in
    /// `solver::persistent_warm_start::persistent_store` — otherwise every
    /// fit reset the counter and ran a full eviction walk on its first save
    /// (gam#1114).
    byte_total: Arc<AtomicU64>,
    /// Monotonically increasing save counter, shared across clones. Used
    /// together with `byte_total` to throttle the eviction directory walk.
    save_counter: Arc<AtomicU64>,
    /// Root-directory mtime observed at the last completed eviction sweep,
    /// shared across clones. When a throttled sweep fires while the store is
    /// comfortably *under* the size budget, the only work left for it is a
    /// TTL/byte resync over every key dir — an N-dir `read_dir` + `stat` walk
    /// that, with thousands of fingerprint dirs in a long CI run, dominates
    /// the per-32-save sweep even after the per-dir listing cache lands (the
    /// residual #1114 walk). The root dir's mtime is bumped by the OS whenever
    /// a key dir is created or removed under it, so an unchanged root mtime
    /// means no key dir was added/dropped since our last sweep; combined with
    /// a comfortably-under-budget byte total, the size-eviction walk is then a
    /// guaranteed no-op and is skipped. TTL expiry of *existing* entries does
    /// not change the root mtime, but it is already performed lazily on every
    /// `lookup_with` and on the next root-changing save, so skipping it here is
    /// behaviour-neutral (no entry the gate skips could be returned stale).
    last_evict_root_mtime: Arc<Mutex<Option<SystemTime>>>,
    /// Per-store test-only monotonic time offset (nanoseconds) added to every
    /// `*_now` reading. Always zero in production. Tests mutate it through
    /// [`Self::test_advance_time`] to simulate elapsed time without
    /// `thread::sleep`. Lives on the store rather than as a process-wide
    /// static so parallel tests with their own stores cannot pollute each
    /// other's clocks — a global clock made `cargo test` non-deterministic
    /// (gam test infra: one test's +1.5s TTL advance was bumping another
    /// test's just-saved entry past its 1s TTL on immediate lookup).
    test_time_offset_ns: AtomicU64,
}

impl Clone for WarmStartStore {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            opts: self.opts.clone(),
            index: Arc::clone(&self.index),
            // Throttle counters are shared across clones so the eviction
            // directory walk stays throttled to every Nth save across the
            // whole process, even though `persistent_store` hands out a fresh
            // clone per save/lookup.
            byte_total: Arc::clone(&self.byte_total),
            save_counter: Arc::clone(&self.save_counter),
            last_evict_root_mtime: Arc::clone(&self.last_evict_root_mtime),
            test_time_offset_ns: AtomicU64::new(self.test_time_offset_ns.load(Ordering::Relaxed)),
        }
    }
}

impl WarmStartStore {
    /// Open (or create) a store rooted at `root`.
    pub fn open(root: PathBuf, opts: StoreOptions) -> Result<Self, StoreError> {
        fs::create_dir_all(&root)?;
        Ok(Self {
            root,
            opts,
            index: Arc::new(Mutex::new(MetadataIndex::default())),
            byte_total: Arc::new(AtomicU64::new(0)),
            save_counter: Arc::new(AtomicU64::new(0)),
            last_evict_root_mtime: Arc::new(Mutex::new(None)),
            test_time_offset_ns: AtomicU64::new(0),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn options(&self) -> &StoreOptions {
        &self.opts
    }

    fn key_dir(&self, key: &Fingerprint) -> PathBuf {
        self.root.join(key.to_hex())
    }

    /// Look up the best entry for `key`, or `None` if no valid entry exists.
    ///
    /// Selection: lowest `objective` first; ties prefer [`EntryKind::Final`]
    /// over [`EntryKind::Checkpoint`], then latest `written_unix_secs`. If
    /// every candidate has `objective = None`, picks the latest write.
    /// Corrupt or schema-mismatched candidates are silently cleaned up and
    /// skipped.
    pub fn lookup(&self, key: &Fingerprint) -> Result<Option<WarmStartEntry>, StoreError> {
        self.lookup_with(key, LookupMode::Best)
    }

    /// Look up the newest valid entry for `key`, or `None` if no valid entry
    /// exists.
    ///
    /// Unlike [`Self::lookup`], this deliberately ignores objective values.
    /// Use this for near-match seed namespaces where entries may come from
    /// different folds, diseases, or row sets, and objective magnitudes are
    /// not comparable. Exact-key resume should keep using [`Self::lookup`].
    pub fn lookup_latest(&self, key: &Fingerprint) -> Result<Option<WarmStartEntry>, StoreError> {
        self.lookup_with(key, LookupMode::Latest)
    }

    fn lookup_with(
        &self,
        key: &Fingerprint,
        mode: LookupMode,
    ) -> Result<Option<WarmStartEntry>, StoreError> {
        let dir = self.key_dir(key);
        if !dir.exists() {
            // A stale in-memory cache entry could outlive its directory if
            // another process evicted us. Drop it so we don't return data
            // for a key whose backing files are gone.
            lookup_cache_invalidate(&LookupCacheKey { fp: *key, mode });
            self.metadata_index_remove_key(key);
            return Ok(None);
        }
        // Fast path: if the same (key, mode) was looked up before and the
        // chosen meta file's mtime is unchanged, return the cached entry
        // without re-reading any JSON or re-checksumming the .bin payload.
        // A separate writer (this process or another) bumps mtime on
        // rename → mismatch → we fall through to the slow path. The TTL
        // cutoff is also re-checked here against `nanos_now()` so a hot
        // poll loop cannot keep returning an expired entry between eviction
        // sweeps (eviction is throttled via `EVICT_EVERY_N_SAVES`).
        let cache_key = LookupCacheKey { fp: *key, mode };
        let now_nanos = self.nanos_now();
        if let Some(hit) = lookup_cache_get(&cache_key) {
            if let Ok(md) = fs::metadata(&hit.meta_path)
                && md.modified().ok() == Some(hit.meta_mtime)
            {
                let expired = self.opts.ttl.as_nanos() > 0
                    && now_nanos.saturating_sub(hit.write_nanos) >= self.opts.ttl.as_nanos();
                if !expired {
                    let entry = self.touch_lookup_hit(&hit.meta_path, hit.entry)?;
                    return Ok(Some(entry));
                }
                lookup_cache_invalidate(&cache_key);
                let bin = hit.meta_path.with_extension("bin");
                fs::remove_file(&hit.meta_path).ok();
                fs::remove_file(&bin).ok();
                // Removing the entry stales any cached directory listing.
                self.metadata_index_remove(&hit.meta_path);
                return Ok(None);
            }
            lookup_cache_invalidate(&cache_key);
        }
        // Resolve all valid entries for this key directory. `scan_key_dir`
        // serves the listing from the per-store directory cache when the dir's
        // mtime is unchanged since the last scan (no re-`read_dir`, no per-file
        // `stat`, no JSON re-parse), and drops TTL-expired / corrupt entries in
        // passing — exactly the syscall storm #1114 traced.
        let mut best: Option<(OnDiskMeta, PathBuf)> = None;
        for scanned in self.scan_key_dir(&dir, now_nanos) {
            let take = match best {
                None => true,
                Some((ref cur, _)) => mode.better(&scanned.meta, cur),
            };
            if take {
                best = Some((scanned.meta, scanned.meta_path));
            }
        }
        let (meta, meta_path) = match best {
            Some(b) => b,
            None => {
                lookup_cache_invalidate(&cache_key);
                return Ok(None);
            }
        };
        let bin_path = meta_path.with_extension("bin");
        let payload = match fs::read(&bin_path) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        // Validate checksum
        if checksum_hex(&payload) != meta.checksum_hex {
            fs::remove_file(&meta_path).ok();
            fs::remove_file(&bin_path).ok();
            lookup_cache_invalidate(&cache_key);
            self.metadata_index_remove(&meta_path);
            return Ok(None);
        }
        let entry = WarmStartEntry {
            payload,
            objective: meta.objective,
            iteration: meta.iteration,
            written_unix_secs: meta.written_unix_secs,
            kind: meta.kind,
        };
        let (meta, entry) = self.touch_lookup_meta(&meta_path, meta, entry)?;
        // Record (meta_path, mtime) → entry so subsequent identical lookups
        // short-circuit until the meta file's mtime changes. The effective
        // activity stamp (post-touch, so it reflects this very access) is
        // cached alongside so the fast path can re-apply the TTL cutoff without
        // re-reading the JSON.
        if let Ok(md) = fs::metadata(&meta_path)
            && let Ok(mtime) = md.modified()
        {
            let write_nanos = meta_activity_nanos(&meta);
            lookup_cache_insert(
                cache_key,
                CachedLookup {
                    meta_path: meta_path.clone(),
                    meta_mtime: mtime,
                    write_nanos,
                    entry: entry.clone(),
                },
            );
        }
        Ok(Some(entry))
    }

    /// Save a new entry with a fresh run-id. Returns the run-id (caller may
    /// hand it to [`Self::save_overwrite`] for periodic in-place updates).
    pub fn save(
        &self,
        key: &Fingerprint,
        payload: &[u8],
        objective: Option<f64>,
        iteration: Option<u64>,
        kind: EntryKind,
    ) -> Result<String, StoreError> {
        let run_id = self.fresh_run_id();
        self.save_overwrite(key, &run_id, payload, objective, iteration, kind)?;
        Ok(run_id)
    }

    /// Save under a specific run-id (overwrites an existing entry with the
    /// same id atomically).
    pub fn save_overwrite(
        &self,
        key: &Fingerprint,
        run_id: &str,
        payload: &[u8],
        objective: Option<f64>,
        iteration: Option<u64>,
        kind: EntryKind,
    ) -> Result<(), StoreError> {
        // Any new write under this key may change which entry wins both
        // `LookupMode::Best` and `LookupMode::Latest`, so drop both cached
        // rows before touching disk. A pure save_overwrite of the same
        // run_id would also bump mtime and self-invalidate, but a save()
        // with a fresh run_id leaves the old meta file unchanged — only
        // explicit invalidation catches that.
        lookup_cache_invalidate(&LookupCacheKey {
            fp: *key,
            mode: LookupMode::Best,
        });
        lookup_cache_invalidate(&LookupCacheKey {
            fp: *key,
            mode: LookupMode::Latest,
        });
        let dir = self.key_dir(key);
        let pid = std::process::id();
        // 1. Compute checksum from payload.
        let checksum = checksum_hex(payload);
        let objective_finite = objective.filter(|o| o.is_finite());
        // The meta's `written_unix_secs`/`written_nanos` are captured INSIDE the
        // write loop — just before the meta_tmp is written, AFTER the bin write
        // has completed. The stored timestamp drives the TTL contract: an
        // entry's clock should start ticking from when the entry becomes
        // (nearly) visible to lookups, not from `save_overwrite`'s entry. On
        // slow disks the bin write + fsync + rename can take longer than the
        // entire TTL window itself (the warm-start test fixture pins TTL=1s
        // while the ext4-backed CI image takes >1s on small writes), so an
        // up-front stamp causes the entry to be classified as expired the
        // moment `save_overwrite` returns. Pushing the stamp past the bin
        // fsync removes that systemic drift from the cost of writing the
        // entry — only the meta fsync + final rename + dir fsync still
        // elapse between the stamp and the entry becoming visible.

        // 3. Write both temp files and atomically rename them into place. The
        //    whole "ensure dir → write temps → rename" sequence is retried once
        //    as a unit on `ErrorKind::NotFound`, because a concurrent process'
        //    `evict_overflow` can `remove_dir` this key dir the instant it
        //    observes it empty (store.rs `evict_overflow`, "Sweep now-empty key
        //    dirs"). That removal races every write step here: it can vanish the
        //    dir after `create_dir_all` but before a temp `File::create`, or
        //    take the dir *and our just-written temps with it* before the
        //    rename, surfacing as `io: No such file or directory (os error 2)`
        //    under parallel CV / bootstrap fitting (gam#868). Retrying the
        //    sequence (not an individual step) is the only correct response: a
        //    bare rename retry can't recover once the source temp was swept with
        //    the dir, so we recreate the dir and rewrite the temps from the
        //    in-memory `payload` / `meta_json` we still hold. A single retry is
        //    sufficient — the eviction window is one `remove_dir` syscall wide —
        //    and a second genuine `NotFound` is propagated as before.
        let nonce = self.nanos_now();
        let bin_final = dir.join(format!("{run_id}.bin"));
        let meta_final = dir.join(format!("{run_id}.json"));
        let mut attempt = 0u8;
        let build_meta_json = |secs: u64, subsec_nanos: u32| -> Result<Vec<u8>, StoreError> {
            let meta = OnDiskMeta {
                schema_version: SCHEMA_VERSION,
                written_unix_secs: secs,
                written_nanos: subsec_nanos,
                objective: objective_finite,
                iteration,
                kind,
                checksum_hex: checksum.clone(),
                payload_bytes: payload.len() as u64,
                accessed: false,
                accessed_unix_secs: 0,
                accessed_nanos: 0,
            };
            Ok(serde_json::to_vec_pretty(&meta)?)
        };
        loop {
            let bin_tmp = dir.join(format!("{run_id}.bin.tmp.{pid}.{nonce}.{attempt}"));
            let meta_tmp = dir.join(format!("{run_id}.json.tmp.{pid}.{nonce}.{attempt}"));
            let stamp_fn = || self.unix_now_parts();
            let build_meta_for_io = |secs: u64, subsec_nanos: u32| -> io::Result<Vec<u8>> {
                build_meta_json(secs, subsec_nanos)
                    .map_err(|e| io::Error::other(format!("meta build: {e:?}")))
            };
            match write_and_promote_entry(&EntryWrite {
                dir: &dir,
                bin_tmp: &bin_tmp,
                meta_tmp: &meta_tmp,
                payload,
                bin_final: &bin_final,
                meta_final: &meta_final,
                stamp_fn: &stamp_fn,
                build_meta_json: &build_meta_for_io,
            }) {
                Ok(()) => break,
                Err(e) if e.kind() == io::ErrorKind::NotFound && attempt == 0 => {
                    // A sibling process' eviction removed the key dir mid-write.
                    // Clean up any partial temps, then retry the whole sequence
                    // once after recreating the dir inside `write_and_promote_entry`.
                    fs::remove_file(&bin_tmp).ok();
                    fs::remove_file(&meta_tmp).ok();
                    attempt += 1;
                    continue;
                }
                Err(e) => {
                    fs::remove_file(&bin_tmp).ok();
                    fs::remove_file(&meta_tmp).ok();
                    fs::remove_file(&bin_final).ok();
                    return Err(StoreError::Io(e));
                }
            }
        }
        // Fsync the containing directory so the rename itself is durable
        // across a power loss / hard crash. fs::File::sync_all on the
        // payload only guarantees the file content reaches disk; without
        // also fsyncing the directory inode, the *rename* (which is what
        // makes the entry visible to lookups) can be lost. Best-effort on
        // platforms where opening a directory for fsync is not supported.
        if let Ok(d) = fs::File::open(&dir) {
            d.sync_all().ok();
        }
        self.metadata_index_upsert(&meta_final, &bin_final).ok();
        // 5. Best-effort eviction; failure here is non-fatal. Throttle the
        // full directory scan: maintain a process-wide approximate byte
        // total and only run eviction when the per-save counter wraps
        // `EVICT_EVERY_N_SAVES` as a drift-resync trigger, or on the very
        // first save (so a fresh process inheriting a populated store root
        // sweeps once). The
        // counter is best-effort: it can drift relative to disk truth
        // because other processes may write/evict, but every triggered
        // sweep resyncs it to ground truth.
        //
        // The counter throttle alone does NOT bound the store: a burst of up
        // to `EVICT_EVERY_N_SAVES - 1` saves between two counter-triggered
        // sweeps can push the footprint arbitrarily far past the budget (e.g.
        // 31 payloads under a budget that fits a handful). Bound it by also
        // sweeping whenever the approximate byte total already exceeds the
        // budget — a single cheap atomic load, so the common under-budget path
        // still walks the directory only every Nth save, while an over-budget
        // total forces the very next save to reclaim it. The eviction resyncs
        // `byte_total` to ground truth, so this fires once per crossing rather
        // than on every subsequent save.
        let approx_added = payload.len() as u64 + APPROX_META_BYTES;
        let new_total = self.byte_total.fetch_add(approx_added, Ordering::Relaxed) + approx_added;
        let n = self.save_counter.fetch_add(1, Ordering::Relaxed);
        if n == 0
            || n.is_multiple_of(EVICT_EVERY_N_SAVES)
            || new_total > self.opts.size_budget_bytes
        {
            self.evict_overflow().ok();
        }
        Ok(())
    }

    /// Drop entries older than TTL, then evict by recorded write-time
    /// ascending until total bytes ≤ `opts.size_budget_bytes`. Idempotent;
    /// safe under concurrent processes (worst case some entries are
    /// double-removed, which is a no-op).
    ///
    /// Sort key is the `(written_unix_secs, written_nanos)` recorded in
    /// each entry's meta, not the filesystem mtime — at second-resolution
    /// mtime, batches of writes within the same second would sort
    /// arbitrarily and could evict the most recent entry.
    pub fn evict_overflow(&self) -> Result<(), StoreError> {
        // Root-mtime short-circuit. A throttled sweep that fires while the
        // approximate byte total is comfortably under budget has no size
        // eviction to do; its only residual work is the TTL/byte resync walk
        // over every key dir. The root mtime is bumped whenever a key dir is
        // created/removed beneath it, so if it is unchanged since our last
        // completed sweep AND we are under budget, no key dir was added or
        // dropped and the size-eviction walk is provably a no-op — skip the
        // N-dir `read_dir`+`stat` storm. (TTL expiry of existing entries does
        // not move the root mtime, but it is already enforced lazily on every
        // `lookup_with` and on the next root-changing save, so the gate cannot
        // surface a stale entry.) This trims the residual #1114 walk in long
        // refit-heavy CI runs where thousands of fingerprint dirs accumulate.
        let current_root_mtime = fs::metadata(&self.root)
            .ok()
            .and_then(|m| m.modified().ok());
        if self.byte_total.load(Ordering::Relaxed) <= self.opts.size_budget_bytes
            && let Some(now_mtime) = current_root_mtime
            && let Ok(last) = self.last_evict_root_mtime.lock()
            && *last == Some(now_mtime)
        {
            return Ok(());
        }
        let read_dir = match fs::read_dir(&self.root) {
            Ok(rd) => rd,
            Err(_) => return Ok(()),
        };
        // Collect (meta_path, bin_path, total_bytes, write_nanos_since_epoch, accessed).
        let mut all: Vec<(PathBuf, PathBuf, u64, u128, bool)> = Vec::new();
        let now_nanos = self.nanos_now();
        for key_dir_entry in read_dir {
            let key_dir = match key_dir_entry {
                Ok(e) => e.path(),
                Err(_) => continue,
            };
            if !key_dir.is_dir() {
                continue;
            }
            // `scan_key_dir` reuses the per-store directory-listing cache when
            // the key dir's mtime is unchanged, so an unchanged dir costs a
            // single `stat` rather than a `read_dir` + per-file `stat` + JSON
            // read of every entry. It also sweeps foreign tmp files and drops
            // corrupt / TTL-expired entries, mirroring the old inline pass.
            let scanned = self.scan_key_dir(&key_dir, now_nanos);
            for entry in &scanned {
                let write_nanos = (entry.meta.written_unix_secs as u128) * 1_000_000_000u128
                    + entry.meta.written_nanos as u128;
                let total_bytes = entry.meta_len + entry.bin_len;
                all.push((
                    entry.meta_path.clone(),
                    entry.bin_path.clone(),
                    total_bytes,
                    write_nanos,
                    entry.meta.accessed,
                ));
            }
            // Sweep now-empty key dirs.
            if scanned.is_empty()
                && fs::read_dir(&key_dir)
                    .map(|mut it| it.next().is_none())
                    .unwrap_or(false)
            {
                fs::remove_dir(&key_dir).ok();
                if let Ok(mut index) = self.index.lock() {
                    index.by_key_dir.remove(&key_dir);
                }
            }
        }
        let total: u64 = all.iter().map(|e| e.2).sum();
        if total <= self.opts.size_budget_bytes {
            // Resync the approximate byte counter even when no eviction was
            // needed. Otherwise the in-memory `byte_total` only grows (it
            // never observes deletions made by sibling processes or
            // expiration sweeps), so after enough saves `new_total` exceeds
            // the budget on every call and triggers a full directory walk
            // on every save instead of every Nth save.
            self.byte_total.store(total, Ordering::Relaxed);
            // Record the root mtime observed by this completed under-budget
            // sweep so a subsequent throttled sweep can short-circuit while
            // the root is unchanged. Re-read after the walk: any key dir the
            // walk removed (empty-dir sweep above) bumps the root mtime, and
            // capturing the post-walk value keeps the gate from skipping a
            // genuinely-changed root on the next call.
            if let (Ok(mut last), Some(m)) = (
                self.last_evict_root_mtime.lock(),
                fs::metadata(&self.root)
                    .ok()
                    .and_then(|m| m.modified().ok()),
            ) {
                *last = Some(m);
            }
            return Ok(());
        }
        all.sort_by(|a, b| {
            a.4.cmp(&b.4)
                .then_with(|| a.3.cmp(&b.3))
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut remaining = total;
        for (meta, bin, bytes, _, _) in all.into_iter() {
            if remaining <= self.opts.size_budget_bytes {
                break;
            }
            fs::remove_file(&meta).ok();
            fs::remove_file(&bin).ok();
            self.metadata_index_remove(&meta);
            remaining = remaining.saturating_sub(bytes);
        }
        // Resync the approximate byte counter to ground truth. Subsequent
        // saves increment from here until the next sweep.
        self.byte_total.store(remaining, Ordering::Relaxed);
        Ok(())
    }
}

/// Ensure the key dir exists, write the `.bin` and `.json` temp files, and
/// atomically rename both into place. Returns the raw `io::Error` (not a
/// `StoreError`) so the caller can branch on `ErrorKind::NotFound` to retry the
/// whole sequence after a concurrent eviction removed the dir mid-write
/// (gam#868). Idempotent across a retry: every path is derived from the caller's
/// stable args and the temps are rewritten from the in-memory payload, so a
/// second pass into a freshly recreated dir produces the same final entry.
///
/// `.bin` is renamed before `.json` so a meta-pointing-to-missing-bin window is
/// impossible on the happy path; a reader that catches `.bin`-missing treats the
/// entry as corrupt and cleans it up.
struct EntryWrite<'a> {
    dir: &'a Path,
    bin_tmp: &'a Path,
    meta_tmp: &'a Path,
    payload: &'a [u8],
    bin_final: &'a Path,
    meta_final: &'a Path,
    /// Read the current wall clock as `(unix_secs, subsec_nanos)`. Called
    /// AFTER the bin write and bin fsync complete (and after the bin rename)
    /// so the recorded write time tracks when the entry actually becomes
    /// (nearly) visible, not when `save_overwrite` was first invoked. On
    /// slow disks the bin fsync can dominate save latency and a pre-write
    /// stamp would burn TTL the caller never sees.
    stamp_fn: &'a dyn Fn() -> (u64, u32),
    /// Build the meta JSON given the captured `(secs, subsec_nanos)`. The
    /// closure folds those values into `OnDiskMeta` and serializes it.
    build_meta_json: &'a dyn Fn(u64, u32) -> io::Result<Vec<u8>>,
}

fn write_and_promote_entry(w: &EntryWrite<'_>) -> io::Result<()> {
    // Recreate the dir up front: on the first attempt this is the original
    // `create_dir_all`; on a retry it re-establishes the dir a sibling
    // process' eviction removed.
    fs::create_dir_all(w.dir)?;
    {
        let mut f = fs::File::create(w.bin_tmp)?;
        f.write_all(w.payload)?;
        f.sync_all().ok();
    }
    // Promote the bin first so a crash between the two renames leaves an
    // orphan .bin (cleaned up by `evict_overflow`) rather than a meta
    // pointing at a missing .bin (which the reader would mark corrupt).
    fs::rename(w.bin_tmp, w.bin_final)?;
    // Stamp the meta AFTER the bin promotion. This is the latest moment the
    // timestamp can still be inlined into the meta JSON. The remaining gap
    // before the entry is visible to lookups is one meta write+fsync + the
    // meta rename + the caller's directory fsync — all bounded, so TTL is
    // measured from a near-visible moment instead of from the entry to
    // `save_overwrite`.
    let (secs, subsec_nanos) = (w.stamp_fn)();
    let meta_json = (w.build_meta_json)(secs, subsec_nanos)?;
    {
        let mut f = fs::File::create(w.meta_tmp)?;
        f.write_all(&meta_json)?;
        f.sync_all().ok();
    }
    if let Err(e) = fs::rename(w.meta_tmp, w.meta_final) {
        // Roll back the bin we just promoted to avoid orphaning it, then
        // surface the error so the caller can retry or fail.
        fs::remove_file(w.bin_final).ok();
        return Err(e);
    }
    Ok(())
}

/// Conservative meta-JSON size used by the throttled save counter. Real
/// meta files run ~250-400 bytes after pretty-printing; overestimating
/// just means the throttle fires slightly earlier, never later.
const APPROX_META_BYTES: u64 = 512;

/// How [`WarmStartStore::lookup_with`] ranks candidate entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LookupMode {
    /// Lowest objective wins; ties to [`entry_better`].
    Best,
    /// Newest write wins; objectives ignored.
    Latest,
}

impl LookupMode {
    fn better(&self, candidate: &OnDiskMeta, current: &OnDiskMeta) -> bool {
        match self {
            LookupMode::Best => entry_better(candidate, current),
            LookupMode::Latest => entry_newer(candidate, current),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LookupCacheKey {
    fp: Fingerprint,
    mode: LookupMode,
}

#[derive(Clone)]
struct CachedLookup {
    meta_path: PathBuf,
    meta_mtime: SystemTime,
    /// Full-precision nanosecond write timestamp from the on-disk meta,
    /// kept alongside `entry.written_unix_secs` so the fast path can apply
    /// the same TTL cutoff as `evict_overflow` without re-reading the JSON.
    write_nanos: u128,
    entry: WarmStartEntry,
}

#[derive(Debug, Default)]
struct MetadataIndex {
    by_meta_path: HashMap<PathBuf, IndexedMeta>,
    /// Per-key-directory cached listing, keyed by the directory's mtime.
    ///
    /// A key dir's mtime is bumped by the OS whenever an entry is created,
    /// renamed, or removed inside it (which is exactly when our entries
    /// change). So a matching `dir_mtime` means the set of `<runid>.{json,bin}`
    /// pairs is byte-for-byte what we scanned last time, letting
    /// [`WarmStartStore::scan_key_dir`] return the cached `Vec<ScannedEntry>`
    /// without a fresh `read_dir` or any per-file `stat`/JSON read. This is
    /// what kills the metadata-syscall storm in repeated `lookup_with` /
    /// `evict_overflow` calls within one fit (gam#1114).
    by_key_dir: HashMap<PathBuf, ScannedDir>,
}

#[derive(Debug, Clone)]
struct IndexedMeta {
    meta_mtime: SystemTime,
    meta_len: u64,
    bin_len: u64,
    meta: OnDiskMeta,
}

impl IndexedMeta {
    fn matches(&self, meta_md: &fs::Metadata, bin_md: &fs::Metadata) -> bool {
        meta_md.modified().ok() == Some(self.meta_mtime)
            && meta_md.len() == self.meta_len
            && bin_md.len() == self.bin_len
    }
}

/// Cached result of scanning one key directory: its mtime at scan time plus
/// the resolved entries. Reused verbatim while the dir's mtime is unchanged.
#[derive(Debug, Clone)]
struct ScannedDir {
    dir_mtime: SystemTime,
    entries: Vec<ScannedEntry>,
}

/// One resolved `(meta, bin)` pair discovered during a key-dir scan. Carries
/// everything both the lookup ranker and the eviction sweep need so neither
/// has to re-`stat` or re-read the files when the dir is unchanged.
#[derive(Debug, Clone)]
struct ScannedEntry {
    meta_path: PathBuf,
    bin_path: PathBuf,
    meta_len: u64,
    bin_len: u64,
    meta_mtime: Option<SystemTime>,
    bin_mtime: Option<SystemTime>,
    meta: OnDiskMeta,
}

impl ScannedEntry {
    fn matches_files(&self, meta_md: &fs::Metadata, bin_md: &fs::Metadata) -> bool {
        meta_md.len() == self.meta_len
            && bin_md.len() == self.bin_len
            && meta_md.modified().ok() == self.meta_mtime
            && bin_md.modified().ok() == self.bin_mtime
    }
}

/// True iff a meta with the given (`secs`, `nanos`) write timestamp is older
/// than `ttl` relative to `now_nanos`. Mirrors the cutoff in
/// [`WarmStartStore::evict_overflow`] so `lookup_with` cannot return an entry
/// that the eviction sweep would have dropped.
/// TTL expiry test. `activity_nanos` is the entry's effective activity stamp
/// (`meta_activity_nanos`: the more recent of creation and last access), so a
/// reused entry's TTL restarts from its last lookup rather than its creation.
const fn meta_expired(activity_nanos: u128, ttl: Duration, now_nanos: u128) -> bool {
    let ttl_nanos = ttl.as_nanos();
    if ttl_nanos == 0 {
        return false;
    }
    now_nanos.saturating_sub(activity_nanos) >= ttl_nanos
}

/// Process-wide in-memory cache for [`WarmStartStore::lookup_with`]. Hot poll
/// loops hit the same (key, mode) repeatedly between writes, so caching the
/// resolved entry behind an mtime check eliminates the per-call directory
/// walk, JSON parse, and SHA-256 recomputation. Mtime mismatch — including
/// writes from a sibling process — invalidates the row and falls back to
/// the full slow path.
fn lookup_cache() -> &'static Mutex<HashMap<LookupCacheKey, CachedLookup>> {
    static CACHE: OnceLock<Mutex<HashMap<LookupCacheKey, CachedLookup>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

const LOOKUP_CACHE_MAX_ENTRIES: usize = 128;
const LOOKUP_CACHE_MAX_BYTES: usize = 256 * 1024 * 1024;

const fn cached_lookup_resident_bytes(value: &CachedLookup) -> usize {
    std::mem::size_of::<CachedLookup>().saturating_add(value.entry.payload.capacity())
}

fn lookup_cache_get(key: &LookupCacheKey) -> Option<CachedLookup> {
    let guard = lookup_cache().lock().ok()?;
    guard.get(key).cloned()
}

fn lookup_cache_insert(key: LookupCacheKey, val: CachedLookup) {
    if let Ok(mut guard) = lookup_cache().lock() {
        let new_bytes = cached_lookup_resident_bytes(&val);
        if new_bytes > LOOKUP_CACHE_MAX_BYTES {
            return;
        }
        let mut resident_bytes: usize = guard.values().map(cached_lookup_resident_bytes).sum();
        if let Some(old) = guard.remove(&key) {
            resident_bytes = resident_bytes.saturating_sub(cached_lookup_resident_bytes(&old));
        }
        while guard.len() >= LOOKUP_CACHE_MAX_ENTRIES
            || resident_bytes.saturating_add(new_bytes) > LOOKUP_CACHE_MAX_BYTES
        {
            let oldest = guard
                .iter()
                .min_by_key(|(_, cached)| cached.write_nanos)
                .map(|(old_key, _)| *old_key);
            let Some(oldest) = oldest else {
                break;
            };
            if let Some(old) = guard.remove(&oldest) {
                resident_bytes = resident_bytes.saturating_sub(cached_lookup_resident_bytes(&old));
            }
        }
        guard.insert(key, val);
    }
}

fn lookup_cache_invalidate(key: &LookupCacheKey) {
    if let Ok(mut guard) = lookup_cache().lock() {
        guard.remove(key);
    }
}

/// Run a full [`WarmStartStore::evict_overflow`] sweep every Nth save. The
/// budget can briefly overshoot by K-1 payloads, which the next sweep
/// reclaims. K=32 keeps the amortized cost negligible on hot checkpoint
/// paths while still bounding worst-case disk drift.
const EVICT_EVERY_N_SAVES: u64 = 32;

fn parse_tmp_pid(name: &str) -> Option<u32> {
    // Names look like "<runid>.bin.tmp.<pid>.<nonce>.<attempt>" or
    // "<runid>.json.tmp.<pid>.<nonce>.<attempt>" (the trailing retry-attempt
    // suffix is irrelevant here — only the first token after ".tmp." is the pid).
    let tail = name.split(".tmp.").nth(1)?;
    let pid_str = tail.split('.').next()?;
    pid_str.parse::<u32>().ok()
}

fn read_meta(path: &Path) -> Result<OnDiskMeta, StoreError> {
    let bytes = fs::read(path)?;
    let parsed: OnDiskMeta = serde_json::from_slice(&bytes)?;
    Ok(parsed)
}

fn entry_better(candidate: &OnDiskMeta, current: &OnDiskMeta) -> bool {
    match (candidate.objective, current.objective) {
        (Some(c), Some(d)) => {
            if (c - d).abs() < 1e-12 {
                match (candidate.kind, current.kind) {
                    (EntryKind::Final, EntryKind::Checkpoint) => true,
                    (EntryKind::Checkpoint, EntryKind::Final) => false,
                    _ => entry_newer(candidate, current),
                }
            } else {
                c < d
            }
        }
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => entry_newer(candidate, current),
    }
}

fn entry_newer(candidate: &OnDiskMeta, current: &OnDiskMeta) -> bool {
    let candidate_stamp = (
        candidate.written_unix_secs,
        candidate.written_nanos,
        candidate_kind_rank(candidate.kind),
    );
    let current_stamp = (
        current.written_unix_secs,
        current.written_nanos,
        candidate_kind_rank(current.kind),
    );
    candidate_stamp > current_stamp
}

const fn candidate_kind_rank(kind: EntryKind) -> u8 {
    match kind {
        EntryKind::Checkpoint => 0,
        EntryKind::Final => 1,
    }
}

fn checksum_hex(payload: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(payload);
    let out = h.finalize();
    let mut s = String::with_capacity(out.len() * 2);
    for b in out.iter() {
        use std::fmt::Write;
        write!(&mut s, "{:02x}", b).expect("writing to String is infallible");
    }
    s
}

impl WarmStartStore {
    fn touch_lookup_hit(
        &self,
        meta_path: &Path,
        entry: WarmStartEntry,
    ) -> Result<WarmStartEntry, StoreError> {
        let meta = read_meta(meta_path)?;
        let (_meta, entry) = self.touch_lookup_meta(meta_path, meta, entry)?;
        Ok(entry)
    }

    fn touch_lookup_meta(
        &self,
        meta_path: &Path,
        mut meta: OnDiskMeta,
        entry: WarmStartEntry,
    ) -> Result<(OnDiskMeta, WarmStartEntry), StoreError> {
        let now = self.nanos_now();
        // Refresh the ACCESS stamp (TTL clock), never the creation stamp: the
        // creation stamp is the ordering key for `lookup_latest`, so bumping it
        // on a read would make a merely-read entry win "latest" over a strictly
        // newer write. Advance strictly past the previous access stamp so a
        // second touch inside the same nanosecond still moves forward.
        let old_access =
            (meta.accessed_unix_secs as u128) * 1_000_000_000u128 + meta.accessed_nanos as u128;
        let touched = now.max(old_access.saturating_add(1));
        meta.accessed_unix_secs = (touched / 1_000_000_000u128) as u64;
        meta.accessed_nanos = (touched % 1_000_000_000u128) as u32;
        meta.accessed = true;
        let json = serde_json::to_vec_pretty(&meta)?;
        let tmp = meta_path.with_extension(format!(
            "json.touch.tmp.{}.{}",
            std::process::id(),
            self.nanos_now()
        ));
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(&json)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, meta_path)?;
        if let Some(dir) = meta_path.parent()
            && let Ok(d) = fs::File::open(dir)
        {
            d.sync_all().ok();
        }
        self.metadata_index_remove(meta_path);
        // `entry.written_unix_secs` intentionally keeps the immutable creation
        // stamp — the touch above only advanced the access clock.
        Ok((meta, entry))
    }

    fn read_meta_indexed(
        &self,
        path: &Path,
        meta_md: &fs::Metadata,
        bin_md: &fs::Metadata,
    ) -> Result<OnDiskMeta, StoreError> {
        if let Ok(index) = self.index.lock()
            && let Some(cached) = index.by_meta_path.get(path)
            && cached.matches(meta_md, bin_md)
        {
            return Ok(cached.meta.clone());
        }

        let meta = read_meta(path)?;
        let Some(meta_mtime) = meta_md.modified().ok() else {
            return Ok(meta);
        };
        if let Ok(mut index) = self.index.lock() {
            index.by_meta_path.insert(
                path.to_path_buf(),
                IndexedMeta {
                    meta_mtime,
                    meta_len: meta_md.len(),
                    bin_len: bin_md.len(),
                    meta: meta.clone(),
                },
            );
        }
        Ok(meta)
    }

    fn metadata_index_upsert(&self, meta_path: &Path, bin_path: &Path) -> Result<(), StoreError> {
        let meta_md = fs::metadata(meta_path)?;
        let bin_md = fs::metadata(bin_path)?;
        self.read_meta_indexed(meta_path, &meta_md, &bin_md)?;
        // A fresh entry just landed in this key dir, so any cached listing for
        // the dir is stale. Drop it; the next scan rebuilds and re-caches.
        if let Some(parent) = meta_path.parent()
            && let Ok(mut index) = self.index.lock()
        {
            index.by_key_dir.remove(parent);
        }
        Ok(())
    }

    fn metadata_index_remove(&self, meta_path: &Path) {
        if let Ok(mut index) = self.index.lock() {
            index.by_meta_path.remove(meta_path);
            if let Some(parent) = meta_path.parent() {
                index.by_key_dir.remove(parent);
            }
        }
    }

    fn metadata_index_remove_key(&self, key: &Fingerprint) {
        let dir = self.key_dir(key);
        if let Ok(mut index) = self.index.lock() {
            index.by_meta_path.retain(|path, _| !path.starts_with(&dir));
            index.by_key_dir.remove(&dir);
        }
    }

    /// Cached listing lookup for one key directory.
    ///
    /// Returns the cached `Vec<ScannedEntry>` if the directory's current mtime
    /// matches the cached scan (no entry added/removed since), otherwise
    /// `None` so the caller performs a fresh scan via [`Self::scan_key_dir`].
    ///
    /// A matching dir mtime guarantees the *set* of files is unchanged, but TTL
    /// is wall-clock relative, so an entry valid at scan time can expire while
    /// the listing is still cached. The caller re-applies the TTL cutoff to the
    /// returned entries; this only proves the file set is stable.
    fn cached_dir_scan(&self, dir: &Path, dir_md: &fs::Metadata) -> Option<Vec<ScannedEntry>> {
        let dir_mtime = dir_md.modified().ok()?;
        let index = self.index.lock().ok()?;
        let cached = index.by_key_dir.get(dir)?;
        if cached.dir_mtime != dir_mtime {
            return None;
        }
        for entry in &cached.entries {
            let meta_md = fs::metadata(&entry.meta_path).ok()?;
            let bin_md = fs::metadata(&entry.bin_path).ok()?;
            if !entry.matches_files(&meta_md, &bin_md) {
                return None;
            }
        }
        Some(cached.entries.clone())
    }

    fn store_dir_scan(&self, dir: &Path, dir_mtime: SystemTime, entries: &[ScannedEntry]) {
        if let Ok(mut index) = self.index.lock() {
            index.by_key_dir.insert(
                dir.to_path_buf(),
                ScannedDir {
                    dir_mtime,
                    entries: entries.to_vec(),
                },
            );
        }
    }

    /// Scan one key directory, resolving every valid `(meta, bin)` pair and
    /// cleaning up corrupt / orphaned / schema-mismatched files in passing.
    ///
    /// Serves both [`Self::lookup_with`] and [`Self::evict_overflow`]: when the
    /// directory's mtime is unchanged since the previous scan it returns the
    /// cached listing without a single `read_dir`, `metadata`, or JSON read —
    /// the metadata-syscall storm that #1114 traced. A fresh scan re-caches the
    /// listing keyed by the dir mtime observed *after* any cleanup, so a later
    /// unchanged call hits the cache. (`now_nanos` drives the TTL drop; expired
    /// entries are removed and excluded from the result.)
    ///
    /// `.tmp.*` files belonging to other processes are swept; same-PID temps
    /// (in-flight writes from us) are left alone.
    fn scan_key_dir(&self, dir: &Path, now_nanos: u128) -> Vec<ScannedEntry> {
        let dir_md = match fs::metadata(dir) {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        if let Some(cached) = self.cached_dir_scan(dir, &dir_md) {
            // The file set is unchanged, but TTL is wall-clock relative: an
            // entry valid when scanned may have expired since. Re-apply the
            // cutoff against `now_nanos`, removing any that crossed it. If none
            // expired we return the cached listing untouched (the fast path);
            // otherwise the removals bump the dir mtime, so we drop the stale
            // cache and re-cache the survivors keyed by the post-removal mtime.
            let any_expired = cached
                .iter()
                .any(|e| meta_expired(meta_activity_nanos(&e.meta), self.opts.ttl, now_nanos));
            if !any_expired {
                return cached;
            }
            let mut survivors = Vec::with_capacity(cached.len());
            for entry in cached {
                if meta_expired(meta_activity_nanos(&entry.meta), self.opts.ttl, now_nanos) {
                    fs::remove_file(&entry.meta_path).ok();
                    fs::remove_file(&entry.bin_path).ok();
                    self.metadata_index_remove(&entry.meta_path);
                } else {
                    survivors.push(entry);
                }
            }
            if let Some(mtime) = fs::metadata(dir).ok().and_then(|m| m.modified().ok()) {
                self.store_dir_scan(dir, mtime, &survivors);
            }
            return survivors;
        }
        let read_dir = match fs::read_dir(dir) {
            Ok(rd) => rd,
            Err(_) => return Vec::new(),
        };
        let mut entries = Vec::new();
        let mut mutated = false;
        for f in read_dir {
            let path = match f {
                Ok(e) => e.path(),
                Err(_) => continue,
            };
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(s) => s,
                None => continue,
            };
            if name.contains(".tmp.") {
                if let Some(pid) = parse_tmp_pid(name)
                    && pid != std::process::id()
                {
                    fs::remove_file(&path).ok();
                    mutated = true;
                }
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let meta_md = match fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let bin = path.with_extension("bin");
            let bin_md = match fs::metadata(&bin) {
                Ok(m) => m,
                Err(_) => {
                    fs::remove_file(&path).ok();
                    self.metadata_index_remove(&path);
                    mutated = true;
                    continue;
                }
            };
            let meta = match self.read_meta_indexed(&path, &meta_md, &bin_md) {
                Ok(m) => m,
                Err(_) => {
                    fs::remove_file(&path).ok();
                    fs::remove_file(&bin).ok();
                    self.metadata_index_remove(&path);
                    mutated = true;
                    continue;
                }
            };
            if meta.schema_version != SCHEMA_VERSION {
                fs::remove_file(&path).ok();
                fs::remove_file(&bin).ok();
                self.metadata_index_remove(&path);
                mutated = true;
                continue;
            }
            if meta_expired(meta_activity_nanos(&meta), self.opts.ttl, now_nanos) {
                fs::remove_file(&path).ok();
                fs::remove_file(&bin).ok();
                self.metadata_index_remove(&path);
                mutated = true;
                continue;
            }
            entries.push(ScannedEntry {
                meta_path: path,
                bin_path: bin,
                meta_len: meta_md.len(),
                bin_len: bin_md.len(),
                meta_mtime: meta_md.modified().ok(),
                bin_mtime: bin_md.modified().ok(),
                meta,
            });
        }
        // Cache keyed by the mtime *after* any cleanup so the next unchanged
        // call is a cache hit. If cleanup mutated the dir, re-stat to capture
        // the post-mutation mtime; otherwise reuse the mtime we already read.
        let final_mtime = if mutated {
            fs::metadata(dir).ok().and_then(|m| m.modified().ok())
        } else {
            dir_md.modified().ok()
        };
        if let Some(mtime) = final_mtime {
            self.store_dir_scan(dir, mtime, &entries);
        }
        entries
    }

    fn test_time_offset_ns(&self) -> u64 {
        self.test_time_offset_ns.load(Ordering::Relaxed)
    }

    fn unix_now_parts(&self) -> (u64, u32) {
        let base = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let total = base.saturating_add(u128::from(self.test_time_offset_ns()));
        let secs = (total / 1_000_000_000u128) as u64;
        let nanos = (total % 1_000_000_000u128) as u32;
        (secs, nanos)
    }

    fn nanos_now(&self) -> u128 {
        let base = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        base.saturating_add(u128::from(self.test_time_offset_ns()))
    }

    fn fresh_run_id(&self) -> String {
        let pid = std::process::id();
        let nanos = self.nanos_now();
        format!("r{pid:x}-{nanos:x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::warm_start::key::Fingerprinter;

    impl WarmStartStore {
        /// Advance this store's simulated monotonic clock by `dur`. Only
        /// available in tests — production code reads the real wall clock and
        /// never mutates the per-store offset.
        fn test_advance_time(&self, dur: Duration) {
            self.test_time_offset_ns
                .fetch_add(dur.as_nanos() as u64, Ordering::Relaxed);
        }
    }

    fn temp_store() -> (tempfile::TempDir, WarmStartStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = WarmStartStore::open(
            dir.path().to_path_buf(),
            StoreOptions {
                size_budget_bytes: 1024 * 1024,
                ttl: Duration::from_secs(60),
            },
        )
        .unwrap();
        (dir, store)
    }

    fn key_for(s: &str) -> Fingerprint {
        let mut fp = Fingerprinter::new();
        fp.absorb_str(b"test", s);
        fp.finalize()
    }

    #[test]
    fn roundtrip_save_then_lookup() {
        let (_d, store) = temp_store();
        let key = key_for("roundtrip");
        store
            .save(
                &key,
                b"hello-warm",
                Some(1.5),
                Some(7),
                EntryKind::Checkpoint,
            )
            .unwrap();
        let got = store.lookup(&key).unwrap().unwrap();
        assert_eq!(got.payload, b"hello-warm");
        assert_eq!(got.objective, Some(1.5));
        assert_eq!(got.iteration, Some(7));
        assert_eq!(got.kind, EntryKind::Checkpoint);
    }

    #[test]
    fn lookup_picks_lowest_objective() {
        let (_d, store) = temp_store();
        let key = key_for("multi");
        store
            .save(&key, b"worse", Some(3.0), Some(1), EntryKind::Checkpoint)
            .unwrap();
        store
            .save(&key, b"better", Some(1.0), Some(2), EntryKind::Checkpoint)
            .unwrap();
        store
            .save(&key, b"mid", Some(2.0), Some(3), EntryKind::Checkpoint)
            .unwrap();
        let got = store.lookup(&key).unwrap().unwrap();
        assert_eq!(got.payload, b"better");
        assert_eq!(got.objective, Some(1.0));
    }

    #[test]
    fn lookup_latest_ignores_objective_ordering() {
        let (_d, store) = temp_store();
        let key = key_for("latest-vs-best");
        store
            .save(&key, b"low-objective", Some(1.0), Some(1), EntryKind::Final)
            .unwrap();
        store.test_advance_time(Duration::from_millis(2));
        store
            .save(
                &key,
                b"newer-higher-objective",
                Some(10.0),
                Some(2),
                EntryKind::Checkpoint,
            )
            .unwrap();

        let best = store.lookup(&key).unwrap().unwrap();
        assert_eq!(best.payload, b"low-objective");

        let latest = store.lookup_latest(&key).unwrap().unwrap();
        assert_eq!(latest.payload, b"newer-higher-objective");
        assert_eq!(latest.iteration, Some(2));
    }

    #[test]
    fn tiebreak_final_beats_checkpoint() {
        let (_d, store) = temp_store();
        let key = key_for("tie");
        store
            .save(&key, b"ckpt", Some(1.0), None, EntryKind::Checkpoint)
            .unwrap();
        // Same objective, different kind.
        store
            .save(&key, b"final", Some(1.0), None, EntryKind::Final)
            .unwrap();
        let got = store.lookup(&key).unwrap().unwrap();
        assert_eq!(got.payload, b"final");
        assert_eq!(got.kind, EntryKind::Final);
    }

    #[test]
    fn tiebreak_latest_mtime_when_no_objective() {
        let (_d, store) = temp_store();
        let key = key_for("latest");
        store
            .save(&key, b"first", None, None, EntryKind::Checkpoint)
            .unwrap();
        store.test_advance_time(Duration::from_millis(1_100));
        store
            .save(&key, b"second", None, None, EntryKind::Checkpoint)
            .unwrap();
        let got = store.lookup(&key).unwrap().unwrap();
        assert_eq!(got.payload, b"second");
    }

    #[test]
    fn corrupt_payload_is_cleaned_up() {
        let (_d, store) = temp_store();
        let key = key_for("corrupt");
        store
            .save(&key, b"original", Some(0.0), None, EntryKind::Checkpoint)
            .unwrap();
        // Tamper with the .bin file.
        let dir = store.key_dir(&key);
        for entry in fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().and_then(|s| s.to_str()) == Some("bin") {
                fs::write(&p, b"tampered!").unwrap();
            }
        }
        let got = store.lookup(&key).unwrap();
        assert!(got.is_none(), "tampered entry must be rejected");
        // The corrupt files should be cleaned up so they don't accumulate.
        let remaining: Vec<_> = fs::read_dir(&dir).unwrap().collect();
        assert!(remaining.is_empty(), "corrupt entry should be removed");
    }

    #[test]
    fn corrupt_meta_json_is_cleaned_up() {
        let (_d, store) = temp_store();
        let key = key_for("badjson");
        store
            .save(&key, b"x", None, None, EntryKind::Checkpoint)
            .unwrap();
        let dir = store.key_dir(&key);
        for entry in fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().and_then(|s| s.to_str()) == Some("json") {
                fs::write(&p, b"{not valid json").unwrap();
            }
        }
        let got = store.lookup(&key).unwrap();
        assert!(got.is_none());
    }

    #[test]
    fn schema_mismatched_entry_is_cleaned_up() {
        let (_d, store) = temp_store();
        let key = key_for("schema");
        store
            .save(&key, b"x", None, None, EntryKind::Checkpoint)
            .unwrap();
        let dir = store.key_dir(&key);
        for entry in fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().and_then(|s| s.to_str()) == Some("json") {
                let raw = fs::read(&p).unwrap();
                let mut parsed: serde_json::Value = serde_json::from_slice(&raw).unwrap();
                parsed["schema_version"] = serde_json::json!(SCHEMA_VERSION + 99);
                fs::write(&p, serde_json::to_vec_pretty(&parsed).unwrap()).unwrap();
            }
        }
        assert!(store.lookup(&key).unwrap().is_none());
        let remaining: Vec<_> = fs::read_dir(&dir).unwrap().collect();
        assert!(
            remaining.is_empty(),
            "schema-mismatched entry should be removed"
        );
    }

    #[test]
    fn schema_mismatched_entry_is_removed_during_save_eviction_path() {
        let dir = tempfile::tempdir().unwrap();
        let store = WarmStartStore::open(
            dir.path().to_path_buf(),
            StoreOptions {
                size_budget_bytes: 6 * 1024,
                ttl: Duration::from_secs(3600),
            },
        )
        .unwrap();
        let stale_key = key_for("schema-size-stale");
        store
            .save(
                &stale_key,
                &vec![0u8; 4 * 1024],
                None,
                None,
                EntryKind::Checkpoint,
            )
            .unwrap();

        let stale_dir = store.key_dir(&stale_key);
        let mut stale_meta = None;
        let mut stale_bin = None;
        for entry in fs::read_dir(&stale_dir).unwrap() {
            let p = entry.unwrap().path();
            match p.extension().and_then(|s| s.to_str()) {
                Some("json") => {
                    let raw = fs::read(&p).unwrap();
                    let mut parsed: serde_json::Value = serde_json::from_slice(&raw).unwrap();
                    parsed["schema_version"] = serde_json::json!(SCHEMA_VERSION + 99);
                    fs::write(&p, serde_json::to_vec_pretty(&parsed).unwrap()).unwrap();
                    stale_meta = Some(p);
                }
                Some("bin") => stale_bin = Some(p),
                _ => {}
            }
        }
        let stale_meta = stale_meta.expect("saved entry should have metadata");
        let stale_bin = stale_bin.expect("saved entry should have payload");

        let fresh_key = key_for("schema-size-fresh");
        store
            .save(
                &fresh_key,
                &vec![1u8; 2 * 1024],
                None,
                None,
                EntryKind::Checkpoint,
            )
            .unwrap();

        assert!(
            !stale_meta.exists(),
            "schema-mismatched metadata should be removed during eviction scan"
        );
        assert!(
            !stale_bin.exists(),
            "schema-mismatched payload should be removed during eviction scan"
        );

        let mut total = 0u64;
        for key_dir in fs::read_dir(store.root()).unwrap() {
            let key_dir = key_dir.unwrap().path();
            if key_dir.is_dir() {
                for entry in fs::read_dir(key_dir).unwrap() {
                    total += fs::metadata(entry.unwrap().path()).unwrap().len();
                }
            }
        }
        assert!(
            total <= store.options().size_budget_bytes,
            "schema-mismatched bytes must not leak past size accounting (got {total})"
        );
        assert!(store.lookup(&stale_key).unwrap().is_none());
        assert!(store.lookup(&fresh_key).unwrap().is_some());
    }

    #[test]
    fn missing_bin_treated_as_missing() {
        let (_d, store) = temp_store();
        let key = key_for("nobin");
        store
            .save(&key, b"x", None, None, EntryKind::Checkpoint)
            .unwrap();
        let dir = store.key_dir(&key);
        for entry in fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().and_then(|s| s.to_str()) == Some("bin") {
                fs::remove_file(&p).unwrap();
            }
        }
        assert!(store.lookup(&key).unwrap().is_none());
    }

    #[test]
    fn missing_key_returns_none() {
        let (_d, store) = temp_store();
        let key = key_for("absent");
        assert!(store.lookup(&key).unwrap().is_none());
    }

    #[test]
    fn lru_eviction_under_size_budget() {
        let dir = tempfile::tempdir().unwrap();
        // Tiny budget: 4 KiB. Each entry payload + meta JSON is ~600 B.
        let store = WarmStartStore::open(
            dir.path().to_path_buf(),
            StoreOptions {
                size_budget_bytes: 4 * 1024,
                ttl: Duration::from_secs(3600),
            },
        )
        .unwrap();
        let mut keys = Vec::new();
        for i in 0..20 {
            let mut fp = Fingerprinter::new();
            fp.absorb_u64(b"i", i);
            let key = fp.finalize();
            keys.push(key);
            let payload = vec![0u8; 256];
            store
                .save(&key, &payload, Some(i as f64), None, EntryKind::Checkpoint)
                .unwrap();
        }
        // Walk the store root and confirm total bytes is bounded.
        let mut total = 0u64;
        for kd in fs::read_dir(store.root()).unwrap() {
            let kd = kd.unwrap().path();
            if kd.is_dir() {
                for f in fs::read_dir(&kd).unwrap() {
                    total += fs::metadata(f.unwrap().path()).unwrap().len();
                }
            }
        }
        assert!(
            total <= 8 * 1024,
            "eviction failed to bound size (got {total})"
        );
        // Earliest keys must have been evicted; latest survive.
        assert!(store.lookup(&keys[0]).unwrap().is_none());
        assert!(store.lookup(keys.last().unwrap()).unwrap().is_some());
    }

    #[test]
    fn ttl_drops_old_entries() {
        // Expiration is driven by `test_advance_time` (additive simulated time
        // on top of the wall clock), so the TTL itself only needs to be larger
        // than any plausible save→lookup wall-time on the CI runner. The
        // 1-second TTL the original fixture used was tighter than the worst
        // ext4 fsync this image sees (see `save_overwrite`'s late-stamp
        // comment), so the first `is_some()` check would flake to "expired"
        // before any time advance ever ran. 60 s clears that race with margin.
        let dir = tempfile::tempdir().unwrap();
        let ttl = Duration::from_secs(60);
        let store = WarmStartStore::open(
            dir.path().to_path_buf(),
            StoreOptions {
                size_budget_bytes: 1024 * 1024,
                ttl,
            },
        )
        .unwrap();
        let key = key_for("ttl");
        store
            .save(&key, b"x", None, None, EntryKind::Checkpoint)
            .unwrap();
        assert!(store.lookup(&key).unwrap().is_some());
        store.test_advance_time(ttl + Duration::from_secs(5));
        // Trigger eviction via a save under an unrelated key.
        let other = key_for("ttl-other");
        store
            .save(&other, b"y", None, None, EntryKind::Checkpoint)
            .unwrap();
        // Original now expired.
        assert!(store.lookup(&key).unwrap().is_none());
        assert!(store.lookup(&other).unwrap().is_some());
    }

    #[test]
    fn orphan_temp_files_from_dead_processes_are_swept() {
        let (_d, store) = temp_store();
        let key = key_for("tmp");
        let dir = store.key_dir(&key);
        fs::create_dir_all(&dir).unwrap();
        // Use PID 1 — never the current process, so it counts as "other".
        let orphan_other = dir.join("r0-0.json.tmp.1.0");
        let mine = dir.join(format!("r0-0.bin.tmp.{}.0", std::process::id()));
        fs::write(&orphan_other, b"orphan").unwrap();
        fs::write(&mine, b"mine").unwrap();
        store.evict_overflow().unwrap();
        assert!(!orphan_other.exists(), "other-PID tmp file should be swept");
        assert!(mine.exists(), "same-PID tmp file must be left alone");
    }

    #[test]
    fn tmp_filenames_without_pid_are_skipped() {
        // Malformed tmp names (no parseable pid) must not crash the sweep.
        let (_d, store) = temp_store();
        let key = key_for("malformed");
        let dir = store.key_dir(&key);
        fs::create_dir_all(&dir).unwrap();
        let weird = dir.join("garbage.tmp.notapid.suffix");
        fs::write(&weird, b"x").unwrap();
        // Must not panic.
        store.evict_overflow().unwrap();
        assert!(weird.exists());
    }

    #[test]
    fn save_overwrite_keeps_single_entry() {
        let (_d, store) = temp_store();
        let key = key_for("overwrite");
        let id = store
            .save(&key, b"v1", Some(2.0), Some(1), EntryKind::Checkpoint)
            .unwrap();
        store
            .save_overwrite(&key, &id, b"v2", Some(1.0), Some(2), EntryKind::Checkpoint)
            .unwrap();
        // Only one (meta, bin) pair on disk.
        let dir = store.key_dir(&key);
        let files: Vec<_> = fs::read_dir(&dir).unwrap().collect();
        assert_eq!(files.len(), 2, "overwrite should not create a new run-id");
        let got = store.lookup(&key).unwrap().unwrap();
        assert_eq!(got.payload, b"v2");
        assert_eq!(got.objective, Some(1.0));
    }

    #[test]
    fn write_and_promote_recreates_dir_removed_before_write() {
        // gam#868: a sibling process' eviction can `remove_dir` the key dir the
        // instant it observes it empty, racing every write step in `save`. The
        // promote helper must recreate the dir rather than failing with ENOENT.
        let (_d, store) = temp_store();
        let key = key_for("race-recreate");
        let dir = store.key_dir(&key);
        // Dir does NOT exist yet (simulates eviction having removed it after a
        // prior `create_dir_all`). The helper must create it and succeed.
        assert!(!dir.exists());
        let bin_tmp = dir.join("r0.bin.tmp.1.0.0");
        let meta_tmp = dir.join("r0.json.tmp.1.0.0");
        let bin_final = dir.join("r0.bin");
        let meta_final = dir.join("r0.json");
        let stamp_fn = || (0u64, 0u32);
        let build_meta_json = |_: u64, _: u32| -> io::Result<Vec<u8>> { Ok(b"{}".to_vec()) };
        write_and_promote_entry(&EntryWrite {
            dir: &dir,
            bin_tmp: &bin_tmp,
            meta_tmp: &meta_tmp,
            payload: b"payload",
            bin_final: &bin_final,
            meta_final: &meta_final,
            stamp_fn: &stamp_fn,
            build_meta_json: &build_meta_json,
        })
        .expect("promote into a missing dir must recreate it and succeed");
        assert!(bin_final.exists() && meta_final.exists());
        assert_eq!(fs::read(&bin_final).unwrap(), b"payload");
    }

    #[test]
    fn save_survives_concurrent_eviction_removing_key_dir() {
        // gam#868 end-to-end: hammer the same key with concurrent saves while a
        // sibling thread repeatedly runs `evict_overflow` (which `remove_dir`s
        // now-empty key dirs). Before the atomic-retry fix, a save whose
        // `create_dir_all`→write/rename window straddled a `remove_dir` failed
        // with `io: No such file or directory (os error 2)`. Every save must now
        // succeed; we assert no save returns an error.
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let dir = tempfile::tempdir().unwrap();
        // Zero size budget forces `evict_overflow` to delete entries (and then
        // sweep the emptied key dir) on essentially every sweep, maximizing the
        // race window.
        let store = Arc::new(
            WarmStartStore::open(
                dir.path().to_path_buf(),
                StoreOptions {
                    size_budget_bytes: 0,
                    ttl: Duration::from_secs(60),
                },
            )
            .unwrap(),
        );
        let key = key_for("concurrent-evict");
        let stop = Arc::new(AtomicBool::new(false));

        let evictor = {
            let store = Arc::clone(&store);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    store.evict_overflow().ok();
                }
            })
        };

        let writers: Vec<_> = (0..4)
            .map(|w| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    for i in 0..200u32 {
                        let payload = format!("w{w}-i{i}");
                        store
                            .save(
                                &key,
                                payload.as_bytes(),
                                Some(i as f64),
                                Some(i as u64),
                                EntryKind::Checkpoint,
                            )
                            .expect("save must not fail with ENOENT under concurrent eviction");
                    }
                })
            })
            .collect();

        for h in writers {
            h.join().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        evictor.join().unwrap();
    }

    #[test]
    fn keys_are_isolated() {
        let (_d, store) = temp_store();
        let a = key_for("a");
        let b = key_for("b");
        store
            .save(&a, b"AAA", Some(1.0), None, EntryKind::Final)
            .unwrap();
        store
            .save(&b, b"BBB", Some(1.0), None, EntryKind::Final)
            .unwrap();
        assert_eq!(store.lookup(&a).unwrap().unwrap().payload, b"AAA");
        assert_eq!(store.lookup(&b).unwrap().unwrap().payload, b"BBB");
    }
}
