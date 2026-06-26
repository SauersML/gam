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

    /// `true` once every chunk of the pass has been submitted.
    pub fn is_complete(&self) -> bool {
        self.frontier == self.n_chunks() && self.pending.is_empty()
    }

    /// Submit the rows of chunk `chunk_index` (shape
    /// `(chunk_rows(chunk_index).len(), border_dim)`).
    ///
    /// Chunks may arrive in **any order**; each may be submitted exactly once.
    /// The per-chunk Gram contribution is computed immediately (each entry a
    /// [`pairwise_sum`] over the chunk's rows, in row order), so the caller's
    /// row buffer can be dropped/remapped right after this returns.
    pub fn submit_chunk(
        &mut self,
        chunk_index: usize,
        rows: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        let n_chunks = self.n_chunks();
        if chunk_index >= n_chunks {
            return Err(format!(
                "StreamingBorderGram: chunk index {chunk_index} out of range (n_chunks = {n_chunks})"
            ));
        }
        if chunk_index < self.frontier || self.pending.contains_key(&chunk_index) {
            return Err(format!(
                "StreamingBorderGram: chunk {chunk_index} was already submitted"
            ));
        }
        let expected_rows = self.chunk_rows(chunk_index).len();
        if rows.nrows() != expected_rows || rows.ncols() != self.border_dim {
            return Err(format!(
                "StreamingBorderGram: chunk {chunk_index} has shape ({}, {}) but expected ({}, {})",
                rows.nrows(),
                rows.ncols(),
                expected_rows,
                self.border_dim
            ));
        }
        let gram = self.chunk_gram(rows);
        self.fold_or_park(chunk_index, gram);
        Ok(())
    }

    /// Submit chunk `chunk_index` as a **precomputed** per-chunk Gram partial
    /// (flattened `k·k` row-major), produced by [`chunk_gram_flat`] over exactly
    /// the rows of [`Self::chunk_rows`]`(chunk_index)`.
    ///
    /// This is the cross-node ingestion seam ([`crate::cross_node`]):
    /// a worker node computes its chunks' partials locally and ships the `k·k`
    /// values; the coordinator folds them through the **same** fixed in-order
    /// cascade as row-level submission, so the result is bit-identical to a
    /// single process having seen all the rows. The validation here is
    /// structural (index range, duplicate, partial length); the *content*
    /// contract — that the partial really is `chunk_gram_flat` of the chunk's
    /// rows — is the producer's, enforced by routing both producers through the
    /// one free function.
    pub fn submit_chunk_gram(&mut self, chunk_index: usize, gram: Vec<f64>) -> Result<(), String> {
        let n_chunks = self.n_chunks();
        if chunk_index >= n_chunks {
            return Err(format!(
                "StreamingBorderGram: chunk index {chunk_index} out of range (n_chunks = {n_chunks})"
            ));
        }
        if chunk_index < self.frontier || self.pending.contains_key(&chunk_index) {
            return Err(format!(
                "StreamingBorderGram: chunk {chunk_index} was already submitted"
            ));
        }
        let kk = self.border_dim * self.border_dim;
        if gram.len() != kk {
            return Err(format!(
                "StreamingBorderGram: chunk {chunk_index} partial has len {} but expected {kk}",
                gram.len()
            ));
        }
        if !gram.iter().all(|v| v.is_finite()) {
            return Err(format!(
                "StreamingBorderGram: chunk {chunk_index} partial contains non-finite entries"
            ));
        }
        self.fold_or_park(chunk_index, gram);
        Ok(())
    }

    /// Fold an accepted chunk partial in-order, or park it in the pending
    /// buffer until the frontier reaches it. Shared tail of the row-level and
    /// gram-level submission paths so both produce identical fold behavior.
    fn fold_or_park(&mut self, chunk_index: usize, gram: Vec<f64>) {
        if chunk_index == self.frontier {
            self.fold_chunk(gram);
            self.frontier += 1;
            // Drain any pending chunks the frontier has now reached.
            while let Some(next) = self.pending.remove(&self.frontier) {
                self.fold_chunk(next);
                self.frontier += 1;
            }
        } else {
            self.pending.insert(chunk_index, gram);
        }
    }

    /// Per-chunk Gram contribution, flattened `k·k` row-major — delegates to
    /// the shared free function [`chunk_gram_flat`] so the in-process and
    /// cross-node producers are the same code path, bit for bit.
    fn chunk_gram(&self, rows: ArrayView2<'_, f64>) -> Vec<f64> {
        chunk_gram_flat(rows)
    }

    /// Fold one in-order chunk partial into the cross-chunk cascade. This is
    /// the `StreamingPairwise` push, applied entry-wise to whole chunk Grams:
    /// sequential accumulation within a [`CROSS_CHUNK_BASE`]-chunk base block
    /// (seeded from the block's first partial), then power-of-two cascade
    /// merges of completed blocks.
    fn fold_chunk(&mut self, gram: Vec<f64>) {
        match self.block_partial.as_mut() {
            None => {
                self.block_partial = Some(gram);
                self.block_len = 1;
            }
            Some(acc) => {
                add_into(acc, &gram);
                self.block_len += 1;
            }
        }
        if self.block_len == CROSS_CHUNK_BASE {
            let block = self
                .block_partial
                .take()
                .expect("block_len == CROSS_CHUNK_BASE implies a live block partial");
            self.block_len = 0;
            self.absorb(CROSS_CHUNK_BASE, block);
        }
    }

    /// Merge a completed subtree partial of the given chunk-count `weight`
    /// into the forest, cascading equal-weight merges — the exact
    /// `StreamingPairwise::absorb` cascade, entry-wise on matrices.
    fn absorb(&mut self, weight: usize, value: Vec<f64>) {
        let mut w = weight;
        let mut v = value;
        while let Some((top_w, _)) = self.forest.last() {
            if *top_w == w {
                let (_, top_v) = self
                    .forest
                    .pop()
                    .expect("forest top exists: just observed by last()");
                // combine(left, right): entry-wise add (commutative bitwise).
                v = {
                    let mut merged = top_v;
                    add_into(&mut merged, &v);
                    merged
                };
                w = w.saturating_mul(2);
            } else {
                break;
            }
        }
        self.forest.push((w, v));
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
/// A streaming row source ([`crate::terms::sae::corpus`]) yields batches whose
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
    /// Next chunk index to submit (in-order by construction).
    next_chunk: usize,
}

impl ChunkAssembler {
    /// New assembler over the same partition parameters as
    /// [`StreamingBorderGram::new`].
    pub fn new(border_dim: usize, n_rows: usize, chunk_size: usize) -> Result<Self, String> {
        Ok(Self {
            gram: StreamingBorderGram::new(border_dim, n_rows, chunk_size)?,
            buffer: Vec::new(),
            next_chunk: 0,
        })
    }

    /// Number of buffered rows not yet folded into a chunk.
    fn buffered_rows(&self) -> usize {
        let k = self.gram.border_dim;
        // Structural invariant: rows enter the buffer only via `push_rows`,
        // which rejects any batch whose width is not `k`, and leave only in
        // `need * k` drains — so the length is always a whole multiple of `k`.
        // Kept as a real (release-surviving) assert rather than a `debug_assert`
        // so a future buffer-maintenance bug cannot silently round the row count
        // down.
        assert!(
            self.buffer.len() % k == 0,
            "ChunkAssembler buffer length {} is not a multiple of border_dim {k}",
            self.buffer.len()
        );
        self.buffer.len() / k
    }

    /// Append a batch of rows (any length, including empty) in stream order,
    /// submitting every chunk the buffer completes.
    pub fn push_rows(&mut self, rows: ArrayView2<'_, f64>) -> Result<(), String> {
        let k = self.gram.border_dim;
        if rows.ncols() != k {
            return Err(format!(
                "ChunkAssembler: batch has {} cols but border_dim is {k}",
                rows.ncols()
            ));
        }
        let n_chunks = self.gram.n_chunks();
        // Rows consumed by completed chunks: the final chunk may be short, so
        // clamp to the declared total.
        let consumed = (self.gram.frontier() * self.gram.chunk_size).min(self.gram.n_rows);
        let total_seen = consumed + self.buffered_rows() + rows.nrows();
        if total_seen > self.gram.n_rows {
            return Err(format!(
                "ChunkAssembler: stream overran the declared row count ({} > {})",
                total_seen, self.gram.n_rows
            ));
        }
        for row in rows.outer_iter() {
            self.buffer.extend(row.iter().copied());
        }
        // Submit every completed chunk in order.
        while self.next_chunk < n_chunks {
            let need = self.gram.chunk_rows(self.next_chunk).len();
            if self.buffered_rows() < need {
                break;
            }
            let chunk: Vec<f64> = self.buffer.drain(..need * k).collect();
            let view = ndarray::ArrayView2::from_shape((need, k), &chunk)
                .map_err(|e| format!("ChunkAssembler: chunk reshape failed: {e}"))?;
            self.gram.submit_chunk(self.next_chunk, view)?;
            self.next_chunk += 1;
        }
        Ok(())
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
        let next_chunk = gram.frontier();
        Ok(Self {
            gram,
            buffer: Vec::new(),
            next_chunk,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Deterministic pseudo-random row matrix keyed purely by index.
    fn planted_rows(n: usize, k: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, k), |(i, j)| {
            let x = (i as f64 + 1.0) * 0.7390851 + (j as f64 + 1.0) * 1.6180339;
            (x.sin() * 43_758.547).fract() * 2.0 - 1.0
        })
    }

    fn accumulate_in_order(
        rows: &Array2<f64>,
        chunk_size: usize,
    ) -> (StreamingBorderGram, Vec<usize>) {
        let acc =
            StreamingBorderGram::new(rows.ncols(), rows.nrows(), chunk_size).expect("accumulator");
        let order: Vec<usize> = (0..acc.n_chunks()).collect();
        (acc, order)
    }

    fn run_with_order(rows: &Array2<f64>, chunk_size: usize, order: &[usize]) -> Array2<f64> {
        let mut acc =
            StreamingBorderGram::new(rows.ncols(), rows.nrows(), chunk_size).expect("accumulator");
        for &j in order {
            let range = acc.chunk_rows(j);
            acc.submit_chunk(j, rows.slice(ndarray::s![range, ..]))
                .expect("submit");
        }
        acc.finish().expect("finish")
    }

    fn assert_bit_identical(a: &Array2<f64>, b: &Array2<f64>, label: &str) {
        assert_eq!(a.dim(), b.dim(), "{label}: shape mismatch");
        for ((idx, x), y) in a.indexed_iter().zip(b.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: entry {idx:?} differs bitwise: {x:?} vs {y:?}"
            );
        }
    }

    #[test]
    fn gram_matches_naive_xtx() {
        let n = 257; // deliberately not a multiple of the chunk size
        let k = 5;
        let rows = planted_rows(n, k);
        let gram = run_with_order(&rows, 16, &(0..17).collect::<Vec<_>>());
        let naive = rows.t().dot(&rows);
        for i in 0..k {
            for j in 0..k {
                let d = (gram[[i, j]] - naive[[i, j]]).abs();
                let scale = naive[[i, j]].abs().max(1.0);
                assert!(
                    d <= 1.0e-12 * scale,
                    "Gram[{i},{j}] = {} vs naive {} (delta {d})",
                    gram[[i, j]],
                    naive[[i, j]]
                );
            }
        }
        // Bitwise symmetry: mirror entries reuse the same product sequence.
        for i in 0..k {
            for j in 0..k {
                assert_eq!(gram[[i, j]].to_bits(), gram[[j, i]].to_bits());
            }
        }
    }

    #[test]
    fn bit_reproducible_across_chunk_submission_orders() {
        // Enough chunks (> CROSS_CHUNK_BASE) to exercise the base-block seal,
        // the power-of-two cascade, AND the trailing short block.
        let n = 2 * CROSS_CHUNK_BASE * 3 + 7; // 775 rows
        let k = 4;
        let chunk_size = 2; // 388 chunks
        let rows = planted_rows(n, k);
        let n_chunks = n.div_ceil(chunk_size);

        let in_order: Vec<usize> = (0..n_chunks).collect();
        let reversed: Vec<usize> = (0..n_chunks).rev().collect();
        // Deterministic stride shuffle (388 is coprime to 129).
        let strided: Vec<usize> = (0..n_chunks).map(|i| (i * 129) % n_chunks).collect();

        let g0 = run_with_order(&rows, chunk_size, &in_order);
        let g1 = run_with_order(&rows, chunk_size, &reversed);
        let g2 = run_with_order(&rows, chunk_size, &strided);

        assert_bit_identical(&g0, &g1, "in-order vs reversed submission");
        assert_bit_identical(&g0, &g2, "in-order vs strided submission");
    }

    #[test]
    fn cross_chunk_association_matches_landed_pairwise_sum() {
        // The cross-chunk cascade must associate per-chunk Gram entries
        // EXACTLY as the landed `pairwise_sum` tree does: for every entry,
        // finish() == pairwise_sum(per-chunk entry values), bit for bit.
        let n = 613;
        let k = 3;
        let chunk_size = 2; // 307 chunks: cascade + trailing block both live
        let rows = planted_rows(n, k);
        let mut acc = StreamingBorderGram::new(k, n, chunk_size).expect("accumulator");
        let n_chunks = acc.n_chunks();
        let mut per_chunk_entries: Vec<Vec<f64>> = vec![Vec::with_capacity(n_chunks); k * k];
        for j in 0..n_chunks {
            let range = acc.chunk_rows(j);
            let chunk = rows.slice(ndarray::s![range, ..]);
            let g = acc.chunk_gram(chunk);
            for (e, vals) in g.iter().zip(per_chunk_entries.iter_mut()) {
                vals.push(*e);
            }
            acc.submit_chunk(j, chunk).expect("submit");
        }
        let gram = acc.finish().expect("finish");
        for a in 0..k {
            for b in 0..k {
                let expected = pairwise_sum(&per_chunk_entries[a * k + b]);
                assert_eq!(
                    gram[[a, b]].to_bits(),
                    expected.to_bits(),
                    "entry ({a},{b}): cascade {} vs pairwise_sum {}",
                    gram[[a, b]],
                    expected
                );
            }
        }
    }

    #[test]
    fn resume_equals_straight_through() {
        let n = 491;
        let k = 4;
        let chunk_size = 3;
        let rows = planted_rows(n, k);
        let (acc, order) = accumulate_in_order(&rows, chunk_size);
        let n_chunks = acc.n_chunks();
        // Straight-through reference.
        let straight = run_with_order(&rows, chunk_size, &order);

        // Interrupted run: submit a mixed-order prefix (so the checkpoint
        // carries a live base-block partial, forest entries, AND pending
        // out-of-order chunks), checkpoint through a serde round-trip, resume,
        // submit the rest.
        let mut first = StreamingBorderGram::new(k, n, chunk_size).expect("accumulator");
        let mut submitted = vec![false; n_chunks];
        // Prefix: chunks 0..60 in order, plus three far-ahead chunks.
        let prefix: Vec<usize> = (0..60).chain([150, 100, 163]).collect();
        for &j in &prefix {
            let range = first.chunk_rows(j);
            first
                .submit_chunk(j, rows.slice(ndarray::s![range, ..]))
                .expect("prefix submit");
            submitted[j] = true;
        }
        assert!(
            !first.pending.is_empty(),
            "fixture must exercise pending out-of-order state"
        );
        let json = serde_json::to_string(&first.checkpoint()).expect("serialize checkpoint");
        drop(first);
        let restored: BorderGramCheckpoint =
            serde_json::from_str(&json).expect("deserialize checkpoint");
        let mut second = StreamingBorderGram::resume(restored).expect("resume");
        for j in 0..n_chunks {
            if submitted[j] {
                continue;
            }
            let range = second.chunk_rows(j);
            second
                .submit_chunk(j, rows.slice(ndarray::s![range, ..]))
                .expect("resumed submit");
        }
        let resumed = second.finish().expect("finish resumed");
        assert_bit_identical(&straight, &resumed, "resume vs straight-through");
    }

    #[test]
    fn rejects_duplicates_missing_chunks_and_bad_shapes() {
        let n = 10;
        let k = 2;
        let chunk_size = 4; // chunks: [0,4), [4,8), [8,10)
        let rows = planted_rows(n, k);
        let mut acc = StreamingBorderGram::new(k, n, chunk_size).expect("accumulator");
        assert_eq!(acc.n_chunks(), 3);

        // Wrong shape (short chunk submitted at a full-chunk index).
        let err = acc
            .submit_chunk(0, rows.slice(ndarray::s![0..3, ..]))
            .expect_err("short chunk must be rejected");
        assert!(err.contains("expected (4, 2)"), "got: {err}");

        acc.submit_chunk(0, rows.slice(ndarray::s![0..4, ..]))
            .expect("chunk 0");
        // Duplicate in-order chunk.
        let err = acc
            .submit_chunk(0, rows.slice(ndarray::s![0..4, ..]))
            .expect_err("duplicate must be rejected");
        assert!(err.contains("already submitted"), "got: {err}");

        // Out-of-order chunk 2 (short trailing chunk, 2 rows), then duplicate.
        acc.submit_chunk(2, rows.slice(ndarray::s![8..10, ..]))
            .expect("chunk 2 out of order");
        let err = acc
            .submit_chunk(2, rows.slice(ndarray::s![8..10, ..]))
            .expect_err("duplicate pending must be rejected");
        assert!(err.contains("already submitted"), "got: {err}");

        // Out-of-range chunk index.
        let err = acc
            .submit_chunk(3, rows.slice(ndarray::s![0..4, ..]))
            .expect_err("out-of-range index must be rejected");
        assert!(err.contains("out of range"), "got: {err}");

        // finish() with chunk 1 missing must fail and name it.
        let err = acc.finish().expect_err("missing chunk must fail finish");
        assert!(
            err.contains("[1]"),
            "missing-chunk message must name chunk 1: {err}"
        );
    }

    #[test]
    fn checkpoint_validation_rejects_corruption() {
        let mut acc = StreamingBorderGram::new(3, 100, 10).expect("accumulator");
        let rows = planted_rows(100, 3);
        acc.submit_chunk(0, rows.slice(ndarray::s![0..10, ..]))
            .expect("chunk 0");
        let good = acc.checkpoint();

        let mut bad = good.clone();
        bad.block_len = 0; // inconsistent with a live block_partial
        assert!(StreamingBorderGram::resume(bad).is_err());

        let mut bad = good.clone();
        if let Some(b) = bad.block_partial.as_mut() {
            b.pop(); // wrong partial length
        }
        assert!(StreamingBorderGram::resume(bad).is_err());

        let mut bad = good.clone();
        bad.pending.push((0, vec![0.0; 9])); // pending below the frontier
        assert!(StreamingBorderGram::resume(bad).is_err());

        let mut bad = good;
        bad.frontier = 99; // beyond n_chunks
        assert!(StreamingBorderGram::resume(bad).is_err());
    }

    #[test]
    fn chunk_assembler_is_batching_invariant() {
        // Whatever batch lengths the I/O layer produces, the assembled Gram is
        // bit-identical to slicing the fixed partition directly.
        let n = 463;
        let k = 4;
        let chunk_size = 16;
        let rows = planted_rows(n, k);
        let direct = {
            let (acc, order) = accumulate_in_order(&rows, chunk_size);
            drop(acc);
            run_with_order(&rows, chunk_size, &order)
        };

        // Misaligned, varying batch lengths (3, 5, 7, 11, 13, cycling).
        let mut asm = ChunkAssembler::new(k, n, chunk_size).expect("assembler");
        let sizes = [3usize, 5, 7, 11, 13];
        let mut at = 0usize;
        let mut s = 0usize;
        while at < n {
            let take = sizes[s % sizes.len()].min(n - at);
            asm.push_rows(rows.slice(ndarray::s![at..at + take, ..]))
                .expect("push");
            at += take;
            s += 1;
        }
        let assembled = asm.finish().expect("finish");
        assert_bit_identical(&direct, &assembled, "direct vs assembled batching");
    }

    #[test]
    fn chunk_assembler_checkpoints_only_at_boundaries_and_resumes() {
        let n = 200;
        let k = 3;
        let chunk_size = 10;
        let rows = planted_rows(n, k);
        let direct = run_with_order(&rows, chunk_size, &(0..20).collect::<Vec<_>>());

        let mut asm = ChunkAssembler::new(k, n, chunk_size).expect("assembler");
        // 7 rows: mid-chunk, no checkpoint available.
        asm.push_rows(rows.slice(ndarray::s![0..7, ..]))
            .expect("push");
        assert!(
            asm.checkpoint().is_none(),
            "mid-chunk checkpoint must be None"
        );
        // Up to row 30: exactly 3 chunks folded, boundary checkpoint.
        asm.push_rows(rows.slice(ndarray::s![7..30, ..]))
            .expect("push");
        let cp = asm.checkpoint().expect("boundary checkpoint");
        assert_eq!(cp.frontier, 3);
        drop(asm);

        // Resume: re-read the stream from row frontier * chunk_size.
        let mut resumed = ChunkAssembler::resume(cp).expect("resume");
        resumed
            .push_rows(rows.slice(ndarray::s![30..n, ..]))
            .expect("push rest");
        let gram = resumed.finish().expect("finish");
        assert_bit_identical(&direct, &gram, "assembler resume vs straight-through");
    }

    #[test]
    fn chunk_assembler_rejects_truncated_and_overrunning_streams() {
        let k = 2;
        let rows = planted_rows(30, k);
        // Truncated: declared 30 rows, stream ends at 25 (mid-chunk).
        let mut asm = ChunkAssembler::new(k, 30, 8).expect("assembler");
        asm.push_rows(rows.slice(ndarray::s![0..25, ..]))
            .expect("push");
        let err = asm.finish().expect_err("truncated stream must fail finish");
        assert!(err.contains("mid-chunk"), "got: {err}");
        // Overrun: more rows than declared.
        let mut asm = ChunkAssembler::new(k, 20, 8).expect("assembler");
        let err = asm
            .push_rows(rows.slice(ndarray::s![0..25, ..]))
            .expect_err("overrun must be rejected");
        assert!(err.contains("overran"), "got: {err}");
    }

    /// Mixed-precision error budget (#973): rows stored `f32` (the shard
    /// format) and accumulated in `f64` must reproduce the all-`f64` border
    /// Gram within a **named tolerance**, entry-wise relative to the Gram's
    /// scale. `f32` storage rounds each value to ~6e-8 relative; products
    /// double that; the deterministic f64 pairwise accumulation adds nothing
    /// material. The budget below is ~100× that floor — tight enough to catch
    /// any f32 accumulation sneaking into the path, loose enough to never
    /// flake on legitimate storage rounding.
    const MIXED_PRECISION_BORDER_RTOL: f64 = 1.0e-5;

    #[test]
    fn f32_storage_f64_accumulation_meets_the_error_budget() {
        let n = 700;
        let k = 5;
        let chunk_size = 32;
        let rows = planted_rows(n, k);
        // The storage path: round every value through f32 (exactly what the
        // shard writer + reader do), then accumulate in f64.
        let stored = rows.mapv(|v| f64::from(v as f32));
        let mut acc = StreamingBorderGram::new(k, n, chunk_size).expect("accumulator");
        for j in 0..acc.n_chunks() {
            let range = acc.chunk_rows(j);
            acc.submit_chunk(j, stored.slice(ndarray::s![range, ..]))
                .expect("submit");
        }
        let mixed = acc.finish().expect("finish");
        // All-f64 reference.
        let exact = rows.t().dot(&rows);
        let scale = exact.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
        for i in 0..k {
            for j in 0..k {
                let d = (mixed[[i, j]] - exact[[i, j]]).abs();
                assert!(
                    d <= MIXED_PRECISION_BORDER_RTOL * scale,
                    "Gram[{i},{j}] mixed-precision delta {d:.3e} exceeds budget \
                     {MIXED_PRECISION_BORDER_RTOL:.0e} × scale {scale:.3e}"
                );
            }
        }
    }

    #[test]
    fn zero_rows_yields_zero_gram() {
        let acc = StreamingBorderGram::new(3, 0, 8).expect("accumulator");
        assert_eq!(acc.n_chunks(), 0);
        assert!(acc.is_complete());
        let gram = acc.finish().expect("finish empty");
        assert_eq!(gram.dim(), (3, 3));
        assert!(gram.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()));
    }
}
