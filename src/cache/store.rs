//! Filesystem store for warm-start cache entries.
//!
//! Each entry is a `(<runid>.json, <runid>.bin)` pair inside a per-key
//! directory. Writes go through a temp-file → fsync → rename sequence so a
//! crash mid-write never leaves a half-written entry visible to readers.
//! Per-entry SHA-256 checksums catch any residual corruption.
//!
//! See [`crate::cache`] for the public API summary.

use crate::cache::key::Fingerprint;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// On-disk schema version. Bump on incompatible format changes; old entries
/// are then ignored at read time and evicted on the next save.
pub(crate) const SCHEMA_VERSION: u32 = 1;

/// Default disk-budget for the whole cache root (~1 GiB).
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
pub struct CachedEntry {
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
    /// Approximate sum of bytes written under `root` by this `WarmStartStore`
    /// instance. Used to throttle the full directory-scanning eviction in
    /// [`Self::save_overwrite`] — see `EVICT_EVERY_N_SAVES`. The counter
    /// resyncs to ground truth after every triggered sweep.
    byte_total: AtomicU64,
    /// Monotonically increasing per-instance save counter. Used together
    /// with `byte_total` to throttle the eviction directory walk.
    save_counter: AtomicU64,
}

impl Clone for WarmStartStore {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            opts: self.opts.clone(),
            // Independent throttle counters per clone are fine: each clone
            // will sweep once on its first save and resync from disk.
            byte_total: AtomicU64::new(self.byte_total.load(Ordering::Relaxed)),
            save_counter: AtomicU64::new(0),
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
            byte_total: AtomicU64::new(0),
            save_counter: AtomicU64::new(0),
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
    pub fn lookup(&self, key: &Fingerprint) -> Result<Option<CachedEntry>, StoreError> {
        self.lookup_with(key, LookupMode::Best)
    }

    /// Look up the newest valid entry for `key`, or `None` if no valid entry
    /// exists.
    ///
    /// Unlike [`Self::lookup`], this deliberately ignores objective values.
    /// Use this for near-match seed namespaces where entries may come from
    /// different folds, diseases, or row sets, and objective magnitudes are
    /// not comparable. Exact-key resume should keep using [`Self::lookup`].
    pub fn lookup_latest(&self, key: &Fingerprint) -> Result<Option<CachedEntry>, StoreError> {
        self.lookup_with(key, LookupMode::Latest)
    }

    fn lookup_with(
        &self,
        key: &Fingerprint,
        mode: LookupMode,
    ) -> Result<Option<CachedEntry>, StoreError> {
        let dir = self.key_dir(key);
        if !dir.exists() {
            // A stale in-memory cache entry could outlive its directory if
            // another process evicted us. Drop it so we don't return data
            // for a key whose backing files are gone.
            lookup_cache_invalidate(&LookupCacheKey { fp: *key, mode });
            return Ok(None);
        }
        // Fast path: if the same (key, mode) was looked up before and the
        // chosen meta file's mtime is unchanged, return the cached entry
        // without re-reading any JSON or re-checksumming the .bin payload.
        // A separate writer (this process or another) bumps mtime on
        // rename → mismatch → we fall through to the slow path.
        let cache_key = LookupCacheKey { fp: *key, mode };
        if let Some(hit) = lookup_cache_get(&cache_key) {
            if let Ok(md) = fs::metadata(&hit.meta_path) {
                if md.modified().ok() == Some(hit.meta_mtime) {
                    return Ok(Some(hit.entry));
                }
            }
            lookup_cache_invalidate(&cache_key);
        }
        let mut best: Option<(OnDiskMeta, PathBuf)> = None;
        for entry in fs::read_dir(&dir)? {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            // Only consume *.json (skip .bin, .tmp.*, etc.)
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            // Skip in-flight temp files: <runid>.json.tmp.<pid>
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map_or(false, |n| n.contains(".tmp."))
            {
                continue;
            }
            let meta = match read_meta(&path) {
                Ok(m) => m,
                Err(_) => {
                    let _ = fs::remove_file(&path);
                    continue;
                }
            };
            if meta.schema_version != SCHEMA_VERSION {
                continue;
            }
            // Ensure the .bin sibling exists; otherwise treat as corrupt.
            let bin = path.with_extension("bin");
            if !bin.exists() {
                let _ = fs::remove_file(&path);
                continue;
            }
            let take = match best {
                None => true,
                Some((ref cur, _)) => mode.better(&meta, cur),
            };
            if take {
                best = Some((meta, path));
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
            let _ = fs::remove_file(&meta_path);
            let _ = fs::remove_file(&bin_path);
            lookup_cache_invalidate(&cache_key);
            return Ok(None);
        }
        let entry = CachedEntry {
            payload,
            objective: meta.objective,
            iteration: meta.iteration,
            written_unix_secs: meta.written_unix_secs,
            kind: meta.kind,
        };
        // Record (meta_path, mtime) → entry so subsequent identical lookups
        // short-circuit until the meta file's mtime changes.
        if let Ok(md) = fs::metadata(&meta_path) {
            if let Ok(mtime) = md.modified() {
                lookup_cache_insert(
                    cache_key,
                    CachedLookup {
                        meta_path: meta_path.clone(),
                        meta_mtime: mtime,
                        entry: entry.clone(),
                    },
                );
            }
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
        let run_id = fresh_run_id();
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
        fs::create_dir_all(&dir)?;
        let pid = std::process::id();
        let nonce = nanos_now();
        let bin_tmp = dir.join(format!("{run_id}.bin.tmp.{pid}.{nonce}"));
        let meta_tmp = dir.join(format!("{run_id}.json.tmp.{pid}.{nonce}"));
        let bin_final = dir.join(format!("{run_id}.bin"));
        let meta_final = dir.join(format!("{run_id}.json"));

        // 1. Write payload bytes.
        {
            let mut f = fs::File::create(&bin_tmp)?;
            f.write_all(payload)?;
            let _ = f.sync_all();
        }
        // 2. Compute checksum from payload.
        let checksum = checksum_hex(payload);
        // 3. Write meta JSON.
        let (secs, subsec_nanos) = unix_now_parts();
        let meta = OnDiskMeta {
            schema_version: SCHEMA_VERSION,
            written_unix_secs: secs,
            written_nanos: subsec_nanos,
            objective,
            iteration,
            kind,
            checksum_hex: checksum,
            payload_bytes: payload.len() as u64,
        };
        {
            let json = serde_json::to_vec_pretty(&meta)?;
            let mut f = fs::File::create(&meta_tmp)?;
            f.write_all(&json)?;
            let _ = f.sync_all();
        }
        // 4. Atomic renames. .bin first so a meta-pointing-to-missing-bin
        // window is impossible on the happy path. A reader that catches
        // .bin-missing treats the entry as corrupt and cleans it up.
        let bin_rename = fs::rename(&bin_tmp, &bin_final);
        if let Err(e) = bin_rename {
            let _ = fs::remove_file(&bin_tmp);
            let _ = fs::remove_file(&meta_tmp);
            return Err(StoreError::Io(e));
        }
        if let Err(e) = fs::rename(&meta_tmp, &meta_final) {
            // Roll back the bin we just promoted to avoid orphaning it.
            let _ = fs::remove_file(&bin_final);
            let _ = fs::remove_file(&meta_tmp);
            return Err(StoreError::Io(e));
        }
        // 5. Best-effort eviction; failure here is non-fatal. Throttle the
        // full directory scan: maintain a process-wide approximate byte
        // total and only run eviction when the total may have exceeded the
        // budget, when the per-save counter wraps `EVICT_EVERY_N_SAVES` as
        // a drift-resync fallback, or on the very first save (so a fresh
        // process inheriting a populated cache root sweeps once). The
        // counter is best-effort: it can drift relative to disk truth
        // because other processes may write/evict, but every triggered
        // sweep resyncs it to ground truth.
        let approx_added = payload.len() as u64 + APPROX_META_BYTES;
        let prev_total = self.byte_total.fetch_add(approx_added, Ordering::Relaxed);
        let new_total = prev_total + approx_added;
        let n = self.save_counter.fetch_add(1, Ordering::Relaxed);
        let over_budget = new_total > self.opts.size_budget_bytes;
        if n == 0 || over_budget || n % EVICT_EVERY_N_SAVES == 0 {
            let _ = self.evict_overflow();
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
        let read_dir = match fs::read_dir(&self.root) {
            Ok(rd) => rd,
            Err(_) => return Ok(()),
        };
        // Collect (meta_path, bin_path, total_bytes, write_nanos_since_epoch).
        let mut all: Vec<(PathBuf, PathBuf, u64, u128)> = Vec::new();
        let now_nanos = nanos_now();
        let ttl_nanos = self.opts.ttl.as_nanos();
        for key_dir_entry in read_dir {
            let key_dir = match key_dir_entry {
                Ok(e) => e.path(),
                Err(_) => continue,
            };
            if !key_dir.is_dir() {
                continue;
            }
            let inner = match fs::read_dir(&key_dir) {
                Ok(rd) => rd,
                Err(_) => continue,
            };
            for f in inner {
                let p = match f {
                    Ok(e) => e.path(),
                    Err(_) => continue,
                };
                let name = match p.file_name().and_then(|s| s.to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                // Sweep tmp files from other processes. Same-PID tmps may
                // be in-flight writes from this very process; leave them.
                if name.contains(".tmp.") {
                    if let Some(pid) = parse_tmp_pid(&name) {
                        if pid != std::process::id() {
                            let _ = fs::remove_file(&p);
                        }
                    }
                    continue;
                }
                if p.extension().and_then(|s| s.to_str()) != Some("json") {
                    continue;
                }
                let meta_md = match fs::metadata(&p) {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                let bin = p.with_extension("bin");
                let bin_md = match fs::metadata(&bin) {
                    Ok(m) => m,
                    Err(_) => {
                        // Orphan meta — clean it up.
                        let _ = fs::remove_file(&p);
                        continue;
                    }
                };
                let meta = match read_meta(&p) {
                    Ok(m) => m,
                    Err(_) => {
                        let _ = fs::remove_file(&p);
                        let _ = fs::remove_file(&bin);
                        continue;
                    }
                };
                let write_nanos = (meta.written_unix_secs as u128) * 1_000_000_000u128
                    + meta.written_nanos as u128;
                if ttl_nanos > 0 && now_nanos.saturating_sub(write_nanos) >= ttl_nanos {
                    let _ = fs::remove_file(&p);
                    let _ = fs::remove_file(&bin);
                    continue;
                }
                let total_bytes = meta_md.len() + bin_md.len();
                all.push((p, bin, total_bytes, write_nanos));
            }
            // Sweep now-empty key dirs.
            if fs::read_dir(&key_dir)
                .map(|mut it| it.next().is_none())
                .unwrap_or(false)
            {
                let _ = fs::remove_dir(&key_dir);
            }
        }
        let total: u64 = all.iter().map(|e| e.2).sum();
        if total <= self.opts.size_budget_bytes {
            return Ok(());
        }
        all.sort_by_key(|e| e.3);
        let mut remaining = total;
        for (meta, bin, bytes, _) in all.into_iter() {
            if remaining <= self.opts.size_budget_bytes {
                break;
            }
            let _ = fs::remove_file(&meta);
            let _ = fs::remove_file(&bin);
            remaining = remaining.saturating_sub(bytes);
        }
        // Resync the approximate byte counter to ground truth. Subsequent
        // saves increment from here until the next sweep.
        self.byte_total.store(remaining, Ordering::Relaxed);
        Ok(())
    }
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
    entry: CachedEntry,
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

fn lookup_cache_get(key: &LookupCacheKey) -> Option<CachedLookup> {
    let guard = lookup_cache().lock().ok()?;
    guard.get(key).cloned()
}

fn lookup_cache_insert(key: LookupCacheKey, val: CachedLookup) {
    if let Ok(mut guard) = lookup_cache().lock() {
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
    // Names look like "<runid>.bin.tmp.<pid>.<nonce>" or
    // "<runid>.json.tmp.<pid>.<nonce>".
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

fn candidate_kind_rank(kind: EntryKind) -> u8 {
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
        let _ = write!(&mut s, "{:02x}", b);
    }
    s
}

fn unix_now_parts() -> (u64, u32) {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| (d.as_secs(), d.subsec_nanos()))
        .unwrap_or((0, 0))
}

fn nanos_now() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn fresh_run_id() -> String {
    let pid = std::process::id();
    let nanos = nanos_now();
    format!("r{pid:x}-{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::key::Fingerprinter;
    use std::thread;

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
        let _id = store
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
        thread::sleep(Duration::from_millis(2));
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
        thread::sleep(Duration::from_millis(1100));
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
    fn schema_mismatch_is_ignored() {
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
        // Walk the cache root and confirm total bytes is bounded.
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
        let dir = tempfile::tempdir().unwrap();
        let store = WarmStartStore::open(
            dir.path().to_path_buf(),
            StoreOptions {
                size_budget_bytes: 1024 * 1024,
                ttl: Duration::from_secs(1),
            },
        )
        .unwrap();
        let key = key_for("ttl");
        store
            .save(&key, b"x", None, None, EntryKind::Checkpoint)
            .unwrap();
        assert!(store.lookup(&key).unwrap().is_some());
        thread::sleep(Duration::from_millis(1500));
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
