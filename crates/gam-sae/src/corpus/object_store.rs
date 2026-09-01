//! Object-store shard streaming with bounded prefetch (#987, extending #973).
//!
//! The mmap reader ([`super::shard_reader::MmapShardSource`]) assumes the whole
//! corpus lives on a local filesystem. A frontier activation corpus
//! (10⁹–10¹¹ rows, hundreds of TB) lives in **object storage**; no node ever
//! holds more than a bounded window of it. This module is the seam that makes
//! that regime a [`CorpusRowSource`] like any other — the SAE term, the ρ
//! cascade, and the streaming border-Gram accumulation are all unchanged
//! consumers.
//!
//! ## The trait, not the SDK
//!
//! [`ObjectStore`] is two methods: list the shard keys, fetch (a range of) one
//! object. gam takes no cloud-SDK dependency; an S3/GCS/Azure backend is a
//! ~20-line implementor on the caller's side. The in-tree
//! [`FsObjectStore`] (a directory of `*.shard` files) is the reference
//! implementation and the test double — and also the honest way to run the
//! object-store code path against a network filesystem mount.
//!
//! ## Determinism contract (inherited, not re-invented)
//!
//! The global row order is pinned by the **lexicographically sorted key list**,
//! exactly as the mmap source pins it by sorted file names: shard-0 rows
//! `0..n0`, then shard-1 rows `0..n1`, … with stable global `row_id`s. Fetch
//! latency, retries, and prefetch depth can never reorder, drop, or duplicate
//! rows; the `(row_id, row)` sequence is byte-identical across runs, fleets,
//! and backends, so warm-start keys ([`super::warm_state`]), subsample hashes
//! ([`super::rho_cascade`]), and the cross-node chunk partition
//! ([`gam_solve::cross_node`]) all agree with a local-disk run.
//!
//! ## Bounded prefetch, never materialize
//!
//! At most [`PREFETCH_SHARDS_AHEAD`] shards beyond the one being drained are
//! resident at a time. `next_batch` serves rows out of the front shard's
//! fetched bytes and tops the window up as shards drain; the corpus is never
//! materialized, and the resident set is bounded by the window regardless of
//! corpus size. (An async/multipart store hides its latency *inside*
//! [`ObjectStore::fetch_range`]; this driver only promises it will never ask
//! for more than the window.)
//!
//! ## Mandatory selectivity at this scale
//!
//! At object-store scale, fitting every row is not on the table: the #987
//! contract is that the fit sees a **designed sample** whose inclusion weights
//! are carried into the likelihood so the criterion stays unbiased (the #973
//! subsample-honesty contract, mechanized by
//! [`gam_solve::row_sampling_measure::RowSamplingMeasure::designed_subsample`]).
//! [`designed_sampling_mandatory`] is the auto-derived predicate drivers
//! consult: above the threshold a full-corpus pass is refused as a default and
//! the designed-sample path is the only sanctioned one — selectivity is a
//! correctness-of-economics requirement here, not an optimization.

use ndarray::Array2;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;

use super::shard_reader::{
    CorpusRowSource, DEFAULT_BATCH_ROWS, DEFAULT_PREFETCH_WINDOW_BYTES, DTYPE_F32, HEADER_LEN,
    RowBatch, SHARD_MAGIC, ShardError,
};

/// Maximum number of fully fetched shards held beyond the one currently being
/// drained. Auto-derived policy, not a knob: one shard of look-ahead hides
/// fetch latency for the deterministic next shard; a second guards against a
/// short shard draining faster than the next fetch completes. Together with
/// the per-shard payload this bounds the resident set independent of corpus
/// size.
pub const PREFETCH_SHARDS_AHEAD: usize = 2;

/// Corpus row count at and above which designed (importance-weighted)
/// subsampling is **mandatory** rather than optional: a fit driver seeing at
/// least this many rows must route through
/// [`gam_solve::row_sampling_measure::RowSamplingMeasure::designed_subsample`] and carry
/// the inclusion weights into the likelihood, instead of attempting a
/// full-corpus exact pass. Auto-derived threshold: 10⁸ rows is where even a
/// single linear pass per outer iteration dominates the entire fit budget and
/// where the #973 cascade's honest-subsample arms stop being an optimization
/// and become the only affordable unbiased estimator.
pub const DESIGNED_SAMPLE_MANDATORY_MIN_ROWS: u64 = 100_000_000;

/// Minimal object-store abstraction: list shard keys, fetch object bytes.
///
/// Implementors must be deterministic in *content* (the same key always yields
/// the same bytes during one pass) but are free to be remote, retried, cached,
/// or parallel inside. `fetch_range` has a correct-by-default implementation
/// over `fetch`; backends with native range reads (HTTP `Range`, `pread`)
/// should override it so header probing does not pull whole objects.
pub trait ObjectStore: Send + Sync {

}

/// Reference [`ObjectStore`]: a local directory of `*.shard` files. Doubles as
/// the test backend and as the adapter for network-filesystem mounts.
pub struct FsObjectStore {
    root: PathBuf,
}

impl FsObjectStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

}

impl ObjectStore for FsObjectStore {

}

/// Parsed `v1` shard header (see [`super::shard_reader`] for the layout).
#[derive(Clone, Debug)]
struct ShardMeta {
    key: String,
    n_rows: usize,
    /// Global id of this shard's first row in the concatenated stream.
    global_row_base: u64,
}

/// One fetched, resident shard: its payload bytes (header stripped) plus which
/// shard of the key sequence it is.
struct ResidentShard {
    /// Index into `ObjectStoreShardSource::shards`.
    shard_idx: usize,
    /// Raw little-endian `f32` payload (`n_rows * p * 4` bytes).
    payload: Vec<u8>,
}

/// A [`CorpusRowSource`] streaming `v1` shards out of an [`ObjectStore`] with a
/// bounded prefetch window. See the module docs for the determinism and
/// residency contracts.
pub struct ObjectStoreShardSource {
    store: Arc<dyn ObjectStore>,
    /// Shard metadata in lexicographic key order — the pinned global row order.
    shards: Vec<ShardMeta>,
    p: usize,
    total_rows: u64,
    batch_rows: usize,
    /// Resident fetched shards, front = the shard currently being drained.
    /// Invariant: contiguous shard indices starting at `cursor_shard`, length
    /// ≤ 1 + [`PREFETCH_SHARDS_AHEAD`].
    window: VecDeque<ResidentShard>,
    /// Index into `shards` of the shard currently being drained.
    cursor_shard: usize,
    /// Local row within `shards[cursor_shard]` to read next.
    cursor_local_row: usize,
}

impl ObjectStoreShardSource {
    /// Open a source over every shard the store lists. Headers are probed with
    /// 32-byte range reads so `total_rows` / `width` are known up front without
    /// fetching any payload.
    pub fn open(store: Arc<dyn ObjectStore>) -> Result<Self, ShardError> {
        let mut keys = store.list_shards()?;
        // Sort by key bytes: deterministic and independent of the order the
        // store returns listings in — the exact discipline of the mmap source.
        keys.sort();
        if keys.is_empty() {
            return Err(ShardError::Empty);
        }
        let mut shards = Vec::with_capacity(keys.len());
        let mut p: Option<usize> = None;
        let mut running_base: u64 = 0;
        for key in keys {
            let header = store.fetch_range(&key, 0, HEADER_LEN)?;
            let (n_rows, shard_p) = parse_header(&key, &header)?;
            match p {
                None => p = Some(shard_p),
                Some(expected) if expected != shard_p => {
                    return Err(ShardError::WidthMismatch {
                        expected,
                        found: shard_p,
                        path: PathBuf::from(&key),
                    });
                }
                Some(_) => {}
            }
            shards.push(ShardMeta {
                key,
                n_rows,
                global_row_base: running_base,
            });
            running_base = running_base.saturating_add(n_rows as u64);
        }
        let p = p.ok_or(ShardError::Empty)?;
        let total_rows = running_base;
        if total_rows == 0 {
            return Err(ShardError::Empty);
        }
        // Same auto-derived batch sizing as the mmap source; additionally cap
        // by the shared prefetch-window byte budget so one batch's rows always
        // fit inside the policy window even for very wide activations.
        let row_bytes = p.max(1) * std::mem::size_of::<f32>();
        let window_rows = (DEFAULT_PREFETCH_WINDOW_BYTES / row_bytes).max(1);
        let batch_rows = DEFAULT_BATCH_ROWS
            .min(total_rows as usize)
            .min(window_rows)
            .max(1);
        Ok(Self {
            store,
            shards,
            p,
            total_rows,
            batch_rows,
            window: VecDeque::new(),
            cursor_shard: 0,
            cursor_local_row: 0,
        })
    }

}

impl CorpusRowSource for ObjectStoreShardSource {

    fn width(&self) -> usize {
        self.p
    }

    fn reset(&mut self) {
        self.cursor_shard = 0;
        self.cursor_local_row = 0;
        self.window.clear();
    }

}

#[cfg(test)]
mod tests {
    use super::super::shard_reader::{MmapShardSource, encode_shard_bytes};
    use super::*;
    use ndarray::array;
    use std::io::Write;
    use std::sync::Mutex;

    fn temp_store(name: &str, shards: &[(&str, Array2<f64>)]) -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "gam-sae-objstore-test-{}-{}",
            std::process::id(),
            name
        ));
        std::fs::create_dir_all(&dir).expect("create store dir");
        for (key, rows) in shards {
            let bytes = encode_shard_bytes(rows.view());
            let mut f = File::create(dir.join(key)).expect("create shard");
            f.write_all(&bytes).expect("write shard");
            f.sync_all().expect("sync shard");
        }
        dir
    }

    fn drain(src: &mut dyn CorpusRowSource) -> (Vec<u64>, Vec<f64>) {
        let mut ids = Vec::new();
        let mut vals = Vec::new();
        while let Some(batch) = src.next_batch().expect("batch") {
            ids.extend(batch.row_ids.iter().copied());
            vals.extend(batch.rows.iter().copied());
        }
        (ids, vals)
    }

    #[test]
    fn object_store_replays_the_mmap_row_sequence_exactly() {
        // The same shard set, read via the object-store source and via the
        // mmap source, must yield byte-identical (row_id, row) sequences —
        // the backend is invisible to every downstream determinism contract.
        let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let b = array![[7.0_f64, 8.0], [9.0, 10.0]];
        let dir = temp_store("parity", &[("a.shard", a), ("b.shard", b)]);

        let store = Arc::new(FsObjectStore::new(dir.clone()));
        let mut remote = ObjectStoreShardSource::open(store).expect("open object-store source");
        let mut local = MmapShardSource::open_dir(&dir).expect("open mmap source");

        assert_eq!(remote.total_rows(), local.total_rows());
        assert_eq!(remote.width(), local.width());
        let (ids_r, vals_r) = drain(&mut remote);
        let (ids_l, vals_l) = drain(&mut local);
        assert_eq!(ids_r, ids_l);
        assert_eq!(
            vals_r.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vals_l.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "object-store rows must be bit-identical to mmap rows"
        );

        // reset() replays the identical sequence.
        remote.reset();
        let (ids_again, vals_again) = drain(&mut remote);
        assert_eq!(ids_again, ids_r);
        assert_eq!(vals_again, vals_r);
        if let Err(err) = std::fs::remove_dir_all(&dir) {
            eprintln!(
                "temp object store cleanup failed for {}: {err}",
                dir.display()
            );
        }
    }

    /// A store wrapper that counts whole-shard fetches and asserts the
    /// bounded-window contract: the source never holds fetch results for more
    /// than `1 + PREFETCH_SHARDS_AHEAD` shards at once (we can't observe its
    /// memory directly, but we can observe that fetches happen lazily in key
    /// order rather than all up front).
    struct CountingStore {
        inner: FsObjectStore,
        payload_fetches: Mutex<Vec<String>>,
    }

    impl ObjectStore for CountingStore {
        fn list_shards(&self) -> Result<Vec<String>, ShardError> {
            self.inner.list_shards()
        }
        fn fetch(&self, key: &str) -> Result<Vec<u8>, ShardError> {
            self.inner.fetch(key)
        }
        fn fetch_range(&self, key: &str, offset: u64, len: usize) -> Result<Vec<u8>, ShardError> {
            if offset as usize >= HEADER_LEN {
                self.payload_fetches
                    .lock()
                    .expect(
                        "payload-fetch log lock is poisoned: a writer panicked while holding it",
                    )
                    .push(key.to_string());
            }
            self.inner.fetch_range(key, offset, len)
        }
    }

    #[test]
    fn prefetch_is_bounded_and_in_key_order() {
        // 6 single-row shards; with PREFETCH_SHARDS_AHEAD = 2 the first batch
        // may trigger at most 3 payload fetches, and fetches arrive in key
        // order.
        let mk = |v: f64| array![[v]];
        let dir = temp_store(
            "bounded",
            &[
                ("s0.shard", mk(0.0)),
                ("s1.shard", mk(1.0)),
                ("s2.shard", mk(2.0)),
                ("s3.shard", mk(3.0)),
                ("s4.shard", mk(4.0)),
                ("s5.shard", mk(5.0)),
            ],
        );
        let store = Arc::new(CountingStore {
            inner: FsObjectStore::new(dir.clone()),
            payload_fetches: Mutex::new(Vec::new()),
        });
        let mut src =
            ObjectStoreShardSource::open(Arc::clone(&store) as Arc<dyn ObjectStore>).expect("open");
        let first = src.next_batch().expect("batch").expect("some");
        assert_eq!(first.row_ids, vec![0]);
        {
            let fetched = store
                .payload_fetches
                .lock()
                .expect("payload-fetch log lock is poisoned: a writer panicked while holding it");
            assert!(
                fetched.len() <= 1 + PREFETCH_SHARDS_AHEAD,
                "first batch fetched {} shard payloads; window allows {}",
                fetched.len(),
                1 + PREFETCH_SHARDS_AHEAD
            );
            let mut sorted = fetched.clone();
            sorted.sort();
            assert_eq!(*fetched, sorted, "payload fetches must be in key order");
        }
        // Draining the rest touches every shard exactly once.
        let (ids, _) = drain(&mut src);
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);
        let fetched = store
            .payload_fetches
            .lock()
            .expect("payload-fetch log lock is poisoned: a writer panicked while holding it");
        assert_eq!(fetched.len(), 6, "each shard payload fetched exactly once");
        if let Err(err) = std::fs::remove_dir_all(&dir) {
            eprintln!(
                "temp object store cleanup failed for {}: {err}",
                dir.display()
            );
        }
    }

    #[test]
    fn mandatory_selectivity_threshold_is_pure_and_monotone() {
        assert!(!designed_sampling_mandatory(0));
        assert!(!designed_sampling_mandatory(
            DESIGNED_SAMPLE_MANDATORY_MIN_ROWS - 1
        ));
        assert!(designed_sampling_mandatory(
            DESIGNED_SAMPLE_MANDATORY_MIN_ROWS
        ));
        assert!(designed_sampling_mandatory(u64::MAX));
    }

    #[test]
    fn width_mismatch_is_rejected() {
        let dir = temp_store(
            "width",
            &[
                ("a.shard", array![[1.0_f64, 2.0]]),
                ("b.shard", array![[3.0_f64]]),
            ],
        );
        let store = Arc::new(FsObjectStore::new(dir.clone()));
        let err = ObjectStoreShardSource::open(store);
        assert!(matches!(err, Err(ShardError::WidthMismatch { .. })));
        if let Err(err) = std::fs::remove_dir_all(&dir) {
            eprintln!(
                "temp object store cleanup failed for {}: {err}",
                dir.display()
            );
        }
    }
}
