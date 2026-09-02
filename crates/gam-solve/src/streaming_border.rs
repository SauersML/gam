//! Streaming, deterministic, out-of-core border-Gram accumulation (#973).
//!
//! Corpus-scale joint fits cannot hold the activation row set in memory: the
//! Schur **border Gram** `G = Σ_n x_n x_nᵀ` (with `x_n ∈ ℝ^k` the row's border
//! coordinates) must be accumulated over fixed-size row **chunks** streamed
//! from disk shards. Because the methodological program (replicate nulls,
//! resumable workflows) rests on determinism, the accumulation here is
//! **bit-reproducible by construction**, not by luck:
//!
//! * The chunk partition is a pure function of `(n_rows, chunk_size)` — chunk
//!   `j` covers rows `[j·chunk_size, min((j+1)·chunk_size, n_rows))`.
//! * Each within-chunk Gram entry is a [`pairwise_sum`] over the chunk's rows
//!   (the already-landed deterministic pairwise tree of
//!   [`gam_linalg::pairwise_reduce`]).
//! * Cross-chunk reduction follows the **same fixed pairwise tree** (the
//!   [`StreamingPairwise`](gam_linalg::pairwise_reduce::StreamingPairwise)
//!   cascade, applied entry-wise to whole chunk Grams): sequential base blocks
//!   of [`CROSS_CHUNK_BASE`] chunk partials, then power-of-two cascade merges.
//!   The tree shape depends only on the chunk count — never on values, device
//!   timing, or thread scheduling. A unit test pins the cross-chunk
//!   association bit-for-bit to [`pairwise_sum`] over the per-chunk entries.
//! * Chunks may be **submitted in any order** (e.g. shards finishing on
//!   different devices at different times): every chunk is keyed by its chunk
//!   index, the in-order fold frontier advances eagerly, and out-of-order
//!   arrivals wait in a pending buffer. The final Gram is a pure function of
//!   the row content alone — identical bits for any submission order.
//!
//! All accumulation buffers are **f64** (the mixed-precision policy of #973:
//! per-row kernels may run f32 upstream, but everything feeding evidence
//! accumulates in f64 — this module exposes no f32 accumulation path at all).
//!
//! The accumulation state — partial Grams (in-order fold forest + pending
//! out-of-order chunk partials) plus the chunk cursor — serializes to a
//! [`BorderGramCheckpoint`] and resumes via [`StreamingBorderGram::resume`],
//! with resume-equals-straight-through guaranteed (and unit-tested) at the
//! bit level.
//!
//! Pure library: no SAE coupling, no flags, no environment variables. Drivers
//! that also need a right-hand side `Σ_n x_n y_n` stack the response columns
//! onto the border coordinates (`[X | Y]`) and read the cross block of the
//! returned Gram; per-row weights `w_n` are pre-scaled into the rows as
//! `√w_n · x_n` by the caller.

use gam_linalg::pairwise_reduce::{BASE_CHUNK, pairwise_sum};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Base-block size of the **cross-chunk** pairwise tree, in chunk partials.
///
/// Pinned to the landed [`BASE_CHUNK`] of
/// [`gam_linalg::pairwise_reduce`] so that the entry-wise association order
/// of the cross-chunk fold is bit-identical to [`pairwise_sum`] over the
/// per-chunk entry values (unit-tested below). A pure compile-time constant:
/// the tree shape never depends on tuning, platform, or runtime conditions.
pub const CROSS_CHUNK_BASE: usize = BASE_CHUNK;

/// Serializable accumulation state of a [`StreamingBorderGram`]: the partial
/// Grams plus the chunk cursor. Writing this to disk after every accepted
/// chunk makes a preempted multi-hour pass resumable instead of restartable;
/// [`StreamingBorderGram::resume`] reconstructs the accumulator with
/// bit-identical future behavior (resume-equals-straight-through).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BorderGramCheckpoint {
    /// Border dimension `k` (columns of every submitted chunk).
    pub border_dim: usize,
    /// Total row count of the full pass.
    pub n_rows: usize,
    /// Fixed chunk size (rows per chunk; the last chunk may be shorter).
    pub chunk_size: usize,
    /// Chunk cursor: number of chunks already folded into the in-order
    /// cascade. Chunk indices `< frontier` are consumed; the next in-order
    /// fold is chunk `frontier`.
    pub frontier: usize,
    /// Sequential partial of the current (unsealed) cross-chunk base block,
    /// flattened `k·k` row-major. `None` iff `block_len == 0`.
    pub block_partial: Option<Vec<f64>>,
    /// Number of chunk partials folded into `block_partial`
    /// (`0..CROSS_CHUNK_BASE`).
    pub block_len: usize,
    /// Completed cascade subtrees: `(weight in chunks, flattened k·k partial)`
    /// with strictly decreasing power-of-two-multiple-of-base weights, bottom
    /// to top — exactly the `StreamingPairwise` forest invariant.
    pub forest: Vec<(usize, Vec<f64>)>,
    /// Out-of-order chunk partials waiting for the frontier to reach them:
    /// `(chunk_index, flattened k·k chunk Gram)`, all indices `> frontier`.
    pub pending: Vec<(usize, Vec<f64>)>,
}

/// Chunked, out-of-core, bit-reproducible border-Gram accumulator.
///
/// Accumulates `G = Σ_n x_n x_nᵀ ∈ ℝ^{k×k}` over `n_rows` rows submitted as
/// fixed-size chunks (any submission order), with f64 accumulation throughout
/// and a deterministic pairwise reduction tree whose shape is a pure function
/// of `(n_rows, chunk_size)`. See the module docs for the determinism
/// contract.
pub struct StreamingBorderGram {
    border_dim: usize,
    n_rows: usize,
    chunk_size: usize,
    /// Next chunk index expected by the in-order cascade fold.
    frontier: usize,
    /// Sequential partial of the current cross-chunk base block.
    block_partial: Option<Vec<f64>>,
    /// Chunk partials folded into `block_partial` so far.
    block_len: usize,
    /// Completed cascade subtrees `(weight in chunks, partial)`.
    forest: Vec<(usize, Vec<f64>)>,
    /// Out-of-order chunk partials keyed by chunk index (all `> frontier`).
    pending: BTreeMap<usize, Vec<f64>>,
}

/// Entry-wise in-place accumulation `acc[i] += rhs[i]`.
///
/// IEEE-754 addition is commutative, so `acc + rhs` and `rhs + acc` are
/// bit-identical; only the *association grouping* matters for reproducibility,
/// and that is fixed by the cascade structure of the caller.
fn add_into(acc: &mut [f64], rhs: &[f64]) {
    for (a, r) in acc.iter_mut().zip(rhs.iter()) {
        *a += *r;
    }
}

/// Deterministic per-chunk Gram contribution, flattened `k·k` row-major, with
/// `k = rows.ncols()`. Entry `(a, b)` is the [`pairwise_sum`] of
/// `x_i[a]·x_i[b]` over the chunk's rows in row order; the symmetric mirror
/// entry reuses the same products in the same order, so the matrix is bitwise
/// symmetric.
///
/// Exposed as a free function so a **remote producer** (a worker node in the
/// cross-node reduction, [`crate::cross_node`]) can compute exactly the
/// partial this accumulator would have computed from the same rows, then ship
/// the `k·k` partial instead of the rows. Bit-identical by construction to the
/// in-process path: [`StreamingBorderGram::submit_chunk`] routes through this
/// same function.
pub fn chunk_gram_flat(rows: ArrayView2<'_, f64>) -> Vec<f64> {
    let k = rows.ncols();
    let r = rows.nrows();
    let mut gram = vec![0.0_f64; k * k];
    let mut products = vec![0.0_f64; r];
    for a in 0..k {
        for b in a..k {
            for (i, p) in products.iter_mut().enumerate() {
                *p = rows[[i, a]] * rows[[i, b]];
            }
            let s = pairwise_sum(&products);
            gram[a * k + b] = s;
            gram[b * k + a] = s;
        }
    }
    gram
}

impl StreamingBorderGram {
    /// Create an empty accumulator for `n_rows` total rows of border dimension
    /// `border_dim`, streamed in chunks of `chunk_size` rows.
    pub fn new(border_dim: usize, n_rows: usize, chunk_size: usize) -> Result<Self, String> {
        if border_dim == 0 {
            return Err("StreamingBorderGram: border_dim must be positive".to_string());
        }
        if chunk_size == 0 {
            return Err("StreamingBorderGram: chunk_size must be positive".to_string());
        }
        Ok(Self {
            border_dim,
            n_rows,
            chunk_size,
            frontier: 0,
            block_partial: None,
            block_len: 0,
            forest: Vec::new(),
            pending: BTreeMap::new(),
        })
    }

    /// Total number of chunks of the pass: `ceil(n_rows / chunk_size)`.
    pub fn n_chunks(&self) -> usize {
        self.n_rows.div_ceil(self.chunk_size)
    }

    /// Row range covered by chunk `chunk_index`:
    /// `[chunk_index·chunk_size, min((chunk_index+1)·chunk_size, n_rows))`.
    /// A pure function of the partition parameters — the caller slices its
    /// shard rows with exactly this range.
    pub fn chunk_rows(&self, chunk_index: usize) -> std::ops::Range<usize> {
        let lo = chunk_index * self.chunk_size;
        let hi = ((chunk_index + 1) * self.chunk_size).min(self.n_rows);
        lo..hi
    }

    /// Number of chunks already consumed by the in-order cascade (the chunk
    /// cursor). Pending out-of-order chunks are not counted.
    pub fn frontier(&self) -> usize {
        self.frontier
    }

    /// Serialize the full accumulation state — partial Grams + chunk cursor —
    /// for checkpointing. [`StreamingBorderGram::resume`] reconstructs an
    /// accumulator whose future behavior is bit-identical to never having
    /// stopped.
    pub fn checkpoint(&self) -> BorderGramCheckpoint {
        BorderGramCheckpoint {
            border_dim: self.border_dim,
            n_rows: self.n_rows,
            chunk_size: self.chunk_size,
            frontier: self.frontier,
            block_partial: self.block_partial.clone(),
            block_len: self.block_len,
            forest: self.forest.clone(),
            pending: self
                .pending
                .iter()
                .map(|(idx, g)| (*idx, g.clone()))
                .collect(),
        }
    }

    /// Reconstruct an accumulator from a checkpoint. Validates the structural
    /// invariants so a corrupted checkpoint is rejected loudly instead of
    /// silently producing a wrong (but plausible-looking) Gram.
    pub fn resume(state: BorderGramCheckpoint) -> Result<Self, String> {
        if state.border_dim == 0 {
            return Err("BorderGramCheckpoint: border_dim must be positive".to_string());
        }
        if state.chunk_size == 0 {
            return Err("BorderGramCheckpoint: chunk_size must be positive".to_string());
        }
        let kk = state.border_dim * state.border_dim;
        let n_chunks = state.n_rows.div_ceil(state.chunk_size);
        if state.frontier > n_chunks {
            return Err(format!(
                "BorderGramCheckpoint: frontier {} exceeds n_chunks {n_chunks}",
                state.frontier
            ));
        }
        if state.block_len >= CROSS_CHUNK_BASE {
            return Err(format!(
                "BorderGramCheckpoint: block_len {} must be < CROSS_CHUNK_BASE {CROSS_CHUNK_BASE}",
                state.block_len
            ));
        }
        if state.block_partial.is_some() != (state.block_len > 0) {
            return Err(
                "BorderGramCheckpoint: block_partial presence inconsistent with block_len"
                    .to_string(),
            );
        }
        if let Some(b) = &state.block_partial {
            if b.len() != kk {
                return Err(format!(
                    "BorderGramCheckpoint: block_partial has len {} but expected {kk}",
                    b.len()
                ));
            }
        }
        for (w, g) in &state.forest {
            if *w == 0 || g.len() != kk {
                return Err(
                    "BorderGramCheckpoint: malformed forest partial (zero weight or wrong len)"
                        .to_string(),
                );
            }
        }
        let mut pending = BTreeMap::new();
        for (idx, g) in state.pending {
            if idx < state.frontier || idx >= n_chunks {
                return Err(format!(
                    "BorderGramCheckpoint: pending chunk index {idx} outside (frontier {}, n_chunks {n_chunks})",
                    state.frontier
                ));
            }
            if g.len() != kk {
                return Err(format!(
                    "BorderGramCheckpoint: pending chunk {idx} partial has len {} but expected {kk}",
                    g.len()
                ));
            }
            if pending.insert(idx, g).is_some() {
                return Err(format!(
                    "BorderGramCheckpoint: duplicate pending chunk index {idx}"
                ));
            }
        }
        Ok(Self {
            border_dim: state.border_dim,
            n_rows: state.n_rows,
            chunk_size: state.chunk_size,
            frontier: state.frontier,
            block_partial: state.block_partial,
            block_len: state.block_len,
            forest: state.forest,
            pending,
        })
    }

    /// Finish the pass, returning the `k×k` border Gram. Errors if any chunk
    /// is missing (out-of-order pending chunks the frontier never reached, or
    /// chunks never submitted). The result is a pure function of the row
    /// content: identical bits for any submission order and for any
    /// checkpoint/resume history.
    pub fn finish(mut self) -> Result<Array2<f64>, String> {
        let n_chunks = self.n_chunks();
        if self.frontier != n_chunks {
            let missing: Vec<usize> = (self.frontier..n_chunks)
                .filter(|idx| !self.pending.contains_key(idx))
                .take(8)
                .collect();
            return Err(format!(
                "StreamingBorderGram: finish() before all chunks were submitted \
                 (frontier {}/{n_chunks}, first missing chunk indices {missing:?})",
                self.frontier
            ));
        }
        // Seal the trailing (short) base block, exactly like
        // `StreamingPairwise::finish`.
        if let Some(tail) = self.block_partial.take() {
            let w = self.block_len;
            self.block_len = 0;
            self.forest.push((w, tail));
        }
        // Fold the forest right-to-left: each parent is
        // combine(left_partial, accumulated_right).
        let k = self.border_dim;
        let mut iter = self.forest.into_iter().rev();
        let flat = match iter.next() {
            None => vec![0.0_f64; k * k],
            Some((_, mut acc)) => {
                for (_, left) in iter {
                    add_into(&mut acc, &left);
                }
                acc
            }
        };
        Array2::from_shape_vec((k, k), flat)
            .map_err(|e| format!("StreamingBorderGram: Gram reshape failed: {e}"))
    }
}

/// Bridges arbitrary-length row batches onto the fixed chunk partition.
///
/// A streaming row source (`gam_sae::corpus`) yields batches whose
/// lengths are set by I/O policy (batch size, shard boundaries) — they do
/// **not** align with the deterministic chunk partition the accumulation tree
/// is keyed on. This assembler buffers incoming rows and submits exact chunks
/// in order, so the resulting Gram is bit-identical to having sliced the
/// partition directly: the batching of the producer can never leak into the
/// bits.
///
/// Checkpointing is exposed **at chunk granularity only**:
/// [`ChunkAssembler::checkpoint`] returns `Some` exactly when the internal
/// buffer is empty (a chunk boundary), because buffered raw rows are not part
/// of the accumulation state contract — a resumed pass re-reads its row
/// stream from the checkpointed chunk cursor
/// ([`StreamingBorderGram::chunk_rows`] of the frontier names the next row).
pub struct ChunkAssembler {
    gram: StreamingBorderGram,
    /// Row-major buffered rows (`buffered_rows × border_dim`), not yet a full
    /// chunk.
    buffer: Vec<f64>,
}

impl ChunkAssembler {
    /// New assembler over the same partition parameters as
    /// [`StreamingBorderGram::new`].
    pub fn new(border_dim: usize, n_rows: usize, chunk_size: usize) -> Result<Self, String> {
        Ok(Self {
            gram: StreamingBorderGram::new(border_dim, n_rows, chunk_size)?,
            buffer: Vec::new(),
        })
    }

    /// Serialize the accumulation state — only at a chunk boundary. `None`
    /// while rows are buffered mid-chunk (checkpoint after the next boundary,
    /// or size batches to the chunk size for checkpoint-every-batch).
    pub fn checkpoint(&self) -> Option<BorderGramCheckpoint> {
        if self.buffer.is_empty() {
            Some(self.gram.checkpoint())
        } else {
            None
        }
    }

    /// Resume an assembler at the chunk boundary a checkpoint names. The
    /// caller re-positions its row stream at row
    /// `checkpoint.frontier * checkpoint.chunk_size` (the partition is pure,
    /// so that index is exact) and replays from there.
    pub fn resume(state: BorderGramCheckpoint) -> Result<Self, String> {
        let gram = StreamingBorderGram::resume(state)?;
        Ok(Self {
            gram,
            buffer: Vec::new(),
        })
    }

    /// Finish the pass. Errors if the stream ended mid-chunk or short of the
    /// declared row count — a truncated stream is rejected loudly, never
    /// folded as a silently shorter corpus.
    pub fn finish(self) -> Result<Array2<f64>, String> {
        if !self.buffer.is_empty() {
            let k = self.gram.border_dim;
            return Err(format!(
                "ChunkAssembler: stream ended mid-chunk with {} buffered rows \
                 (declared n_rows = {})",
                self.buffer.len() / k,
                self.gram.n_rows
            ));
        }
        self.gram.finish()
    }
}

