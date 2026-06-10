//! mmap-backed activation-shard reader with bounded prefetch (#973).
//!
//! # On-disk shard format (`v1`)
//!
//! A shard is a single file holding a fixed header followed by a row-major
//! payload. All integers are little-endian. The header is exactly 32 bytes:
//!
//! ```text
//! offset  size  field
//! 0       8     magic            = SHARD_MAGIC  (b"GAMSAE01")
//! 8       8     n_rows  (u64)    number of activation rows in this shard
//! 16      8     p       (u64)    columns per row (activation width)
//! 24      4     dtype   (u32)    DTYPE_F32 = 0  (the only payload encoding)
//! 28      4     reserved(u32)    = 0  (header padding to 32 bytes)
//! 32      ..    payload          n_rows * p contiguous little-endian f32
//! ```
//!
//! The reader memory-maps the file read-only and reads rows directly out of the
//! mapped payload, upcasting each `f32` lane to `f64` on demand (the
//! mixed-precision-storage contract — see [`super::kernels`]). `f32` storage
//! halves on-disk and page-cache footprint versus `f64` so an
//! out-of-core corpus streams without ever materializing as dense `f64`.
//!
//! # Determinism contract
//!
//! [`CorpusRowSource`] yields batches in a **fixed global row order**
//! (shard-0 rows `0..n0`, then shard-1 rows `0..n1`, …), assigning each row a
//! stable global `row_id`. This order is *independent of OS readahead*: the
//! bounded prefetch below only touches pages we are about to read in that same
//! deterministic order — it never reorders, drops, or duplicates rows, and the
//! sequence of `(row_id, row)` pairs is byte-identical across runs and
//! platforms. That stable `row_id` is what [`super::warm_state`] keys its
//! per-row warm starts on and what [`super::rho_cascade`] hashes to pick a
//! subsample.
//!
//! # Bounded prefetch
//!
//! `prefetch_window_bytes` caps how far ahead of the current read cursor the
//! reader hints the OS to fault pages in. This keeps the resident set bounded
//! (we do not want the kernel pulling an entire multi-GiB shard into RAM) while
//! still hiding fault latency for the rows we are about to consume next. The
//! hint is advisory: on platforms without `madvise` we simply touch the first
//! byte of each upcoming page to warm it. Correctness never depends on it.

use memmap2::Mmap;
use ndarray::Array2;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Magic bytes identifying a `v1` activation shard.
pub const SHARD_MAGIC: [u8; 8] = *b"GAMSAE01";
/// `dtype` tag for an `f32` row-major payload (the only supported encoding).
pub const DTYPE_F32: u32 = 0;
/// Fixed header length in bytes.
pub const HEADER_LEN: usize = 32;

/// Default bounded read-ahead window. Auto-derived; not a CLI knob. Large
/// enough to hide fault latency for a healthy batch, small enough that the
/// resident set stays bounded regardless of shard size. Shared with the
/// object-store source ([`super::object_store`]) so both backends apply the
/// same bounded-prefetch policy.
pub(super) const DEFAULT_PREFETCH_WINDOW_BYTES: usize = 8 * 1024 * 1024;

/// Default number of rows handed back per [`CorpusRowSource::next_batch`].
/// Auto-derived from the activation width at open time; this is the floor.
/// Shared with [`super::object_store`].
pub(super) const DEFAULT_BATCH_ROWS: usize = 1024;

#[derive(Debug)]
pub enum ShardError {
    Io(std::io::Error),
    BadMagic {
        path: PathBuf,
    },
    BadDtype {
        path: PathBuf,
        tag: u32,
    },
    Truncated {
        path: PathBuf,
        expected: usize,
        actual: usize,
    },
    /// Two shards in one source disagree on activation width `p`.
    WidthMismatch {
        expected: usize,
        found: usize,
        path: PathBuf,
    },
    /// After a window fill the front resident shard does not match the read
    /// cursor. The window is contractually a contiguous run of shard indices
    /// starting at `cursor_shard`, so the front must equal it; a mismatch is an
    /// internal window-maintenance logic error. Surfaced (rather than silently
    /// reading a payload from one shard against another shard's row metadata)
    /// so corruption never reaches a returned batch.
    ResidencyInvariant {
        cursor_shard: usize,
        front_shard: usize,
    },
    Empty,
}

impl std::fmt::Display for ShardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShardError::Io(e) => write!(f, "shard I/O error: {e}"),
            ShardError::BadMagic { path } => {
                write!(f, "shard '{}' has wrong magic header", path.display())
            }
            ShardError::BadDtype { path, tag } => write!(
                f,
                "shard '{}' has unsupported dtype tag {tag} (only f32={DTYPE_F32})",
                path.display()
            ),
            ShardError::Truncated {
                path,
                expected,
                actual,
            } => write!(
                f,
                "shard '{}' is truncated: header expects {expected} bytes, file has {actual}",
                path.display()
            ),
            ShardError::WidthMismatch {
                expected,
                found,
                path,
            } => write!(
                f,
                "shard '{}' has width p={found}, expected p={expected}",
                path.display()
            ),
            ShardError::ResidencyInvariant {
                cursor_shard,
                front_shard,
            } => write!(
                f,
                "shard window residency invariant violated: read cursor is at shard {cursor_shard} but the window front is shard {front_shard}"
            ),
            ShardError::Empty => write!(f, "shard source has no shards / no rows"),
        }
    }
}

impl std::error::Error for ShardError {}

impl From<std::io::Error> for ShardError {
    fn from(e: std::io::Error) -> Self {
        ShardError::Io(e)
    }
}

/// One deterministic batch of activation rows.
///
/// `row_ids[k]` is the stable global id of `rows.row(k)`; ids are contiguous
/// within a batch and strictly increasing across the whole stream.
#[derive(Debug, Clone)]
pub struct RowBatch {
    /// `(batch_rows × p)` upcast `f64` activations.
    pub rows: Array2<f64>,
    /// Global row id of each row in `rows`, same length as `rows.nrows()`.
    pub row_ids: Vec<u64>,
}

impl RowBatch {
    #[inline]
    pub fn len(&self) -> usize {
        self.row_ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }
}

/// A deterministic, restartable source of activation row batches.
///
/// This is one half of the seam ([`super::warm_state::RowWarmCache`] is the
/// other) that the streaming SAE term will consume. The contract:
///
/// * `next_batch` yields rows in a **fixed global order** with stable
///   `row_id`s, independent of OS readahead, until the corpus is exhausted
///   (then `Ok(None)`).
/// * `reset` rewinds to the first row so a new outer ρ pass replays the exact
///   same `(row_id, row)` sequence — the every-step-is-a-full-corpus-pass
///   contract [`super::rho_cascade`] relies on.
/// * `total_rows` / `width` are known up front (from shard headers) so callers
///   can size accumulators before the first read.
pub trait CorpusRowSource {
    /// Total rows across every shard in this source.
    fn total_rows(&self) -> u64;
    /// Activation width `p` (columns per row).
    fn width(&self) -> usize;
    /// Yield the next deterministic batch, or `Ok(None)` at end of corpus.
    fn next_batch(&mut self) -> Result<Option<RowBatch>, ShardError>;
    /// Rewind to the first row so the next `next_batch` replays from the start.
    fn reset(&mut self);
    /// Rows handed back per `next_batch` (may be smaller for the final batch).
    fn batch_rows(&self) -> usize;
}

/// A single memory-mapped shard.
struct MappedShard {
    mmap: Arc<Mmap>,
    n_rows: usize,
    p: usize,
    data_offset: usize,
    /// Global id of this shard's first row in the concatenated stream.
    global_row_base: u64,
}

impl MappedShard {
    fn open(path: PathBuf) -> Result<Self, ShardError> {
        let file = File::open(&path)?;
        // SAFETY: shards are read-only training artifacts; this module never
        // mutates the mapping and the caller's contract is no concurrent
        // writers. Matches the existing `PcaScoresMemmapDesignOperator` usage.
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < HEADER_LEN {
            return Err(ShardError::Truncated {
                path,
                expected: HEADER_LEN,
                actual: mmap.len(),
            });
        }
        if mmap[0..8] != SHARD_MAGIC {
            return Err(ShardError::BadMagic { path });
        }
        let n_rows = u64::from_le_bytes(mmap[8..16].try_into().expect("8 bytes")) as usize;
        let p = u64::from_le_bytes(mmap[16..24].try_into().expect("8 bytes")) as usize;
        let dtype = u32::from_le_bytes(mmap[24..28].try_into().expect("4 bytes"));
        if dtype != DTYPE_F32 {
            return Err(ShardError::BadDtype { path, tag: dtype });
        }
        let payload_bytes = n_rows
            .checked_mul(p)
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| ShardError::Truncated {
                path: path.clone(),
                expected: usize::MAX,
                actual: mmap.len(),
            })?;
        let expected = HEADER_LEN + payload_bytes;
        if mmap.len() < expected {
            return Err(ShardError::Truncated {
                path,
                expected,
                actual: mmap.len(),
            });
        }
        Ok(Self {
            mmap: Arc::new(mmap),
            n_rows,
            p,
            data_offset: HEADER_LEN,
            global_row_base: 0,
        })
    }

    /// Read a single row's `p` `f32` lanes, upcasting each to `f64`.
    #[inline]
    fn read_row_into(&self, local_row: usize, out: &mut [f64]) {
        assert_eq!(out.len(), self.p);
        let byte_start = self.data_offset + local_row * self.p * std::mem::size_of::<f32>();
        let bytes = &self.mmap[byte_start..byte_start + self.p * std::mem::size_of::<f32>()];
        for (c, slot) in out.iter_mut().enumerate() {
            let b = c * std::mem::size_of::<f32>();
            let lane = f32::from_le_bytes(bytes[b..b + 4].try_into().expect("4 bytes"));
            *slot = f64::from(lane);
        }
    }

    /// Bounded read-ahead: warm pages from `byte_start` up to (but not past)
    /// `byte_start + window`, clamped to the shard payload. Advisory only.
    fn prefetch(&self, byte_start: usize, window: usize) {
        let payload_end = self.data_offset + self.n_rows * self.p * std::mem::size_of::<f32>();
        let end = byte_start.saturating_add(window).min(payload_end);
        if end <= byte_start {
            return;
        }
        // Touch one byte per page so the kernel faults exactly the bounded
        // window in our deterministic read order, never the whole shard via
        // speculative readahead. `read_volatile` keeps the loads from being
        // optimized away without mutating the read-only mapping.
        let page = 4096usize;
        let base = self.mmap.as_ptr();
        let mut off = byte_start;
        while off < end {
            // SAFETY: `off < end <= mmap.len()`, so `base.add(off)` is in
            // bounds of the live read-only mapping; we only read.
            unsafe {
                std::ptr::read_volatile(base.add(off));
            }
            off += page;
        }
    }
}

/// A [`CorpusRowSource`] over one or many shards with a bounded prefetch
/// window.
pub struct MmapShardSource {
    shards: Vec<MappedShard>,
    p: usize,
    total_rows: u64,
    batch_rows: usize,
    prefetch_window_bytes: usize,
    /// Index into `shards` of the shard currently being drained.
    cursor_shard: usize,
    /// Local row within `shards[cursor_shard]` to read next.
    cursor_local_row: usize,
}

impl MmapShardSource {
    /// Open a source over an explicit, ordered list of shard paths. The order
    /// of `paths` *defines* the deterministic global row order, so the caller
    /// must pass a stable ordering (e.g. lexicographically sorted filenames).
    pub fn open(paths: &[PathBuf]) -> Result<Self, ShardError> {
        if paths.is_empty() {
            return Err(ShardError::Empty);
        }
        let mut shards = Vec::with_capacity(paths.len());
        let mut p: Option<usize> = None;
        let mut running_base: u64 = 0;
        for path in paths {
            let mut shard = MappedShard::open(path.clone())?;
            match p {
                None => p = Some(shard.p),
                Some(expected) if expected != shard.p => {
                    return Err(ShardError::WidthMismatch {
                        expected,
                        found: shard.p,
                        path: path.clone(),
                    });
                }
                Some(_) => {}
            }
            shard.global_row_base = running_base;
            running_base = running_base.saturating_add(shard.n_rows as u64);
            shards.push(shard);
        }
        let p = p.ok_or(ShardError::Empty)?;
        let total_rows = running_base;
        if total_rows == 0 {
            return Err(ShardError::Empty);
        }
        // Auto-derive a batch row count: at least DEFAULT_BATCH_ROWS rows, and
        // never more than the whole corpus. No CLI knob.
        let batch_rows = DEFAULT_BATCH_ROWS.min(total_rows as usize).max(1);
        Ok(Self {
            shards,
            p,
            total_rows,
            batch_rows,
            prefetch_window_bytes: DEFAULT_PREFETCH_WINDOW_BYTES,
            cursor_shard: 0,
            cursor_local_row: 0,
        })
    }

    /// Open a source over every `*.shard` file in `dir`, ordered by file name
    /// (the stable, OS-independent ordering that pins the deterministic global
    /// row sequence).
    pub fn open_dir(dir: &Path) -> Result<Self, ShardError> {
        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("shard") {
                paths.push(path);
            }
        }
        // Sort by file name bytes: deterministic and independent of the order
        // the OS returns directory entries in.
        paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
        if paths.is_empty() {
            return Err(ShardError::Empty);
        }
        Self::open(&paths)
    }

    /// True once every row of every shard has been yielded.
    #[inline]
    fn at_end(&self) -> bool {
        self.cursor_shard >= self.shards.len()
    }

    /// Advance the cursor past any fully-drained trailing shards so
    /// `cursor_shard` either points at a shard with remaining rows or is at
    /// `shards.len()` (end of corpus).
    fn skip_drained_shards(&mut self) {
        while self.cursor_shard < self.shards.len()
            && self.cursor_local_row >= self.shards[self.cursor_shard].n_rows
        {
            self.cursor_shard += 1;
            self.cursor_local_row = 0;
        }
    }
}

impl CorpusRowSource for MmapShardSource {
    fn total_rows(&self) -> u64 {
        self.total_rows
    }

    fn width(&self) -> usize {
        self.p
    }

    fn batch_rows(&self) -> usize {
        self.batch_rows
    }

    fn reset(&mut self) {
        self.cursor_shard = 0;
        self.cursor_local_row = 0;
    }

    fn next_batch(&mut self) -> Result<Option<RowBatch>, ShardError> {
        self.skip_drained_shards();
        if self.at_end() {
            return Ok(None);
        }
        // A batch never crosses a shard boundary: it stays within the current
        // shard and is the smaller of the configured batch size and that
        // shard's remaining rows. This keeps row reads contiguous in one
        // mapping and keeps the prefetch window inside one shard's payload.
        let shard_idx = self.cursor_shard;
        let take = {
            let shard = &self.shards[shard_idx];
            let remaining = shard.n_rows - self.cursor_local_row;
            self.batch_rows.min(remaining)
        };

        // Bounded prefetch over exactly the rows we are about to read, in the
        // same deterministic order, before touching them.
        {
            let shard = &self.shards[shard_idx];
            let first_byte =
                shard.data_offset + self.cursor_local_row * shard.p * std::mem::size_of::<f32>();
            let want = take * shard.p * std::mem::size_of::<f32>();
            shard.prefetch(first_byte, want.min(self.prefetch_window_bytes));
        }

        let p = self.p;
        let mut rows = Array2::<f64>::zeros((take, p));
        let mut row_ids = Vec::with_capacity(take);
        {
            let shard = &self.shards[shard_idx];
            for k in 0..take {
                let local = self.cursor_local_row + k;
                let mut row_view = rows.row_mut(k);
                let slice = row_view
                    .as_slice_mut()
                    .expect("freshly allocated contiguous row");
                shard.read_row_into(local, slice);
                row_ids.push(shard.global_row_base + local as u64);
            }
        }
        self.cursor_local_row += take;
        self.skip_drained_shards();
        Ok(Some(RowBatch { rows, row_ids }))
    }
}

/// Serialize a `(n_rows × p)` `f64` activation matrix to the `v1` shard byte
/// layout, downcasting each value to `f32`.
///
/// This is the writer counterpart to [`MmapShardSource`] — used by ingestion
/// tooling and by the round-trip tests below to prove the reader reproduces the
/// exact rows it was given (up to the `f32` storage rounding the format
/// promises). Rows are written in row-major order.
pub fn encode_shard_bytes(rows: ndarray::ArrayView2<'_, f64>) -> Vec<u8> {
    let n_rows = rows.nrows();
    let p = rows.ncols();
    let mut out = Vec::with_capacity(HEADER_LEN + n_rows * p * std::mem::size_of::<f32>());
    out.extend_from_slice(&SHARD_MAGIC);
    out.extend_from_slice(&(n_rows as u64).to_le_bytes());
    out.extend_from_slice(&(p as u64).to_le_bytes());
    out.extend_from_slice(&DTYPE_F32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for row in rows.outer_iter() {
        for &v in row.iter() {
            out.extend_from_slice(&(v as f32).to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::io::Write;

    fn write_temp_shard(name: &str, rows: ndarray::ArrayView2<'_, f64>) -> PathBuf {
        let bytes = encode_shard_bytes(rows);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "gam-sae-corpus-test-{}-{}.shard",
            std::process::id(),
            name
        ));
        let mut f = File::create(&path).expect("create temp shard");
        f.write_all(&bytes).expect("write shard");
        f.sync_all().expect("sync shard");
        path
    }

    #[test]
    fn single_shard_round_trips_rows_and_ids() {
        let data = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let path = write_temp_shard("single", data.view());
        let mut src = MmapShardSource::open(&[path.clone()]).expect("open");
        assert_eq!(src.total_rows(), 3);
        assert_eq!(src.width(), 3);
        let batch = src.next_batch().expect("batch").expect("some");
        assert_eq!(batch.row_ids, vec![0, 1, 2]);
        assert_eq!(batch.rows, data);
        assert!(src.next_batch().expect("end").is_none());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn multi_shard_global_ids_are_contiguous() {
        let a = array![[1.0_f64], [2.0]];
        let b = array![[3.0_f64], [4.0], [5.0]];
        let pa = write_temp_shard("multi-a", a.view());
        let pb = write_temp_shard("multi-b", b.view());
        let mut src = MmapShardSource::open(&[pa.clone(), pb.clone()]).expect("open");
        assert_eq!(src.total_rows(), 5);
        let mut all_ids = Vec::new();
        let mut all_vals = Vec::new();
        while let Some(batch) = src.next_batch().expect("batch") {
            all_ids.extend(batch.row_ids.iter().copied());
            for r in batch.rows.outer_iter() {
                all_vals.push(r[0]);
            }
        }
        assert_eq!(all_ids, vec![0, 1, 2, 3, 4]);
        assert_eq!(all_vals, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        std::fs::remove_file(&pa).ok();
        std::fs::remove_file(&pb).ok();
    }

    #[test]
    fn reset_replays_identical_sequence() {
        let data = array![[1.0_f64, 1.0], [2.0, 2.0]];
        let path = write_temp_shard("reset", data.view());
        let mut src = MmapShardSource::open(&[path.clone()]).expect("open");
        let first: Vec<u64> = {
            let mut ids = Vec::new();
            while let Some(b) = src.next_batch().expect("b") {
                ids.extend(b.row_ids);
            }
            ids
        };
        src.reset();
        let second: Vec<u64> = {
            let mut ids = Vec::new();
            while let Some(b) = src.next_batch().expect("b") {
                ids.extend(b.row_ids);
            }
            ids
        };
        assert_eq!(first, second);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "gam-sae-corpus-badmagic-{}.shard",
            std::process::id()
        ));
        let mut f = File::create(&path).expect("create");
        f.write_all(&[0u8; 64]).expect("write");
        f.sync_all().ok();
        let err = MmapShardSource::open(&[path.clone()]);
        assert!(matches!(err, Err(ShardError::BadMagic { .. })));
        std::fs::remove_file(&path).ok();
    }
}
