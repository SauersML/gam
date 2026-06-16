//! Cross-node deterministic border-Gram reduction (#987, extending #973).
//!
//! [`crate::solver::streaming_border`] made the in-process accumulation of the
//! Schur border Gram `G = Σ_n x_n x_nᵀ` bit-reproducible by construction: the
//! chunk partition is a pure function of `(n_rows, chunk_size)`, per-chunk
//! partials are deterministic [`chunk_gram_flat`] reductions, and the
//! cross-chunk fold is a fixed pairwise tree keyed by **chunk index** — never by
//! arrival order, thread timing, or device count. This module extends that same
//! fixed-shape-by-construction discipline **one level up**, to a fleet of
//! worker nodes, with three properties the frontier corpus regime
//! (10⁹–10¹¹ tokens, hundreds of TB of activations) demands:
//!
//! 1. **Node count never changes bits.** A node's partials are not leaves of a
//!    *separate* per-node tree that then gets merged (that shape would depend
//!    on the node count). Instead every node computes the **globally indexed**
//!    per-chunk partials it owns and ships `(chunk_index, k·k partial)`
//!    messages; the coordinator folds them through the *single* global cascade
//!    of [`StreamingBorderGram`], which accepts any arrival order and folds in
//!    chunk-index order. The reduction topology is therefore a pure function of
//!    `(n_rows, chunk_size)` alone — running on 1 node, 3 nodes, or 64 nodes
//!    yields the identical bit pattern, because the tree never saw the node
//!    count. (The chunk→node *assignment* is rank-indexed and deterministic,
//!    but it only decides who computes a partial, never how partials combine.)
//! 2. **Checkpoint/resume is the job model, not an afterthought.** Any worker's
//!    death resumes from its serialized [`NodeWorkerCheckpoint`] (a cursor into
//!    its owned chunk sequence); the coordinator's full state — the in-order
//!    fold forest, the pending out-of-order partials, and the per-rank receipt
//!    cursors — serializes to a [`CrossNodeCheckpoint`]. Resume-equals-
//!    straight-through holds at the bit level on both sides because both
//!    cursors are positions in deterministic sequences.
//! 3. **Partials, never rows, cross the wire.** A worker streams its shard rows
//!    locally (object store / mmap — [`crate::terms::sae::corpus`]) and ships
//!    only `k·k` f64 partials. The coordinator's ingest seam is
//!    [`StreamingBorderGram::submit_chunk_gram`]; both producers route through
//!    the one [`chunk_gram_flat`] free function, so a shipped partial is
//!    bit-identical to the partial the coordinator would have computed from the
//!    same rows.
//!
//! ## Chunk→rank assignment
//!
//! Round-robin by chunk index: rank `r` of `n_ranks` owns chunks
//! `{j : j ≡ r (mod n_ranks)}`, in increasing order. Round-robin (rather than
//! contiguous ranges) keeps the coordinator's in-order fold frontier advancing
//! steadily while all ranks make progress at similar rates, which bounds the
//! pending out-of-order buffer by O(`n_ranks` × inter-node skew) instead of
//! O(total chunks). The assignment is a pure function of
//! `(chunk_index, n_ranks)`; no scheduler, no work stealing — work stealing
//! would not change bits (the fold is index-keyed) but it *would* break the
//! one-cursor-per-rank resume model, so it is deliberately absent.
//!
//! Pure library: no networking, no flags, no environment variables. The
//! transport (MPI, gRPC, files on a shared filesystem) is the caller's; this
//! module owns the deterministic topology, the cursors, and the validation.

use crate::solver::streaming_border::{BorderGramCheckpoint, StreamingBorderGram, chunk_gram_flat};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

/// The deterministic chunk partition + rank-indexed assignment shared by every
/// participant of one cross-node pass. A pure function of its four fields; two
/// participants constructed with the same fields agree on every derived
/// quantity, with no communication.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossNodePartition {
    /// Border dimension `k` (columns of every chunk; partials are `k·k`).
    pub border_dim: usize,
    /// Total row count of the full pass.
    pub n_rows: usize,
    /// Fixed chunk size (rows per chunk; the last chunk may be shorter).
    pub chunk_size: usize,
    /// Number of worker ranks in the fleet.
    pub n_ranks: usize,
}

impl CrossNodePartition {
    pub fn new(
        border_dim: usize,
        n_rows: usize,
        chunk_size: usize,
        n_ranks: usize,
    ) -> Result<Self, String> {
        if border_dim == 0 {
            return Err("CrossNodePartition: border_dim must be positive".to_string());
        }
        if chunk_size == 0 {
            return Err("CrossNodePartition: chunk_size must be positive".to_string());
        }
        if n_ranks == 0 {
            return Err("CrossNodePartition: n_ranks must be positive".to_string());
        }
        Ok(Self {
            border_dim,
            n_rows,
            chunk_size,
            n_ranks,
        })
    }

    /// Total number of chunks of the pass: `ceil(n_rows / chunk_size)`.
    /// Identical to [`StreamingBorderGram::n_chunks`] for the same partition
    /// parameters — the global tree this assignment feeds.
    pub fn n_chunks(&self) -> usize {
        self.n_rows.div_ceil(self.chunk_size)
    }

    /// Row range covered by global chunk `chunk_index` — the same pure function
    /// as [`StreamingBorderGram::chunk_rows`], duplicated here so a worker can
    /// slice its rows without constructing a coordinator-side accumulator.
    pub fn chunk_rows(&self, chunk_index: usize) -> std::ops::Range<usize> {
        let lo = chunk_index * self.chunk_size;
        let hi = ((chunk_index + 1) * self.chunk_size).min(self.n_rows);
        lo..hi
    }

    /// Which rank owns global chunk `chunk_index`: round-robin by index.
    #[inline]
    pub fn owner_rank(&self, chunk_index: usize) -> usize {
        chunk_index % self.n_ranks
    }

    /// Number of chunks rank `rank` owns.
    pub fn chunks_owned_by(&self, rank: usize) -> usize {
        let n = self.n_chunks();
        if rank >= self.n_ranks || n == 0 {
            return 0;
        }
        // Chunks r, r + n_ranks, r + 2·n_ranks, … below n.
        if rank < n {
            (n - rank - 1) / self.n_ranks + 1
        } else {
            0
        }
    }

    /// The `ordinal`-th (0-based) global chunk index owned by `rank`, or `None`
    /// past the end of the rank's sequence. The worker cursor is an ordinal
    /// into exactly this sequence.
    pub fn owned_chunk(&self, rank: usize, ordinal: usize) -> Option<usize> {
        if rank >= self.n_ranks {
            return None;
        }
        let idx = rank + ordinal * self.n_ranks;
        if idx < self.n_chunks() {
            Some(idx)
        } else {
            None
        }
    }
}

/// One shipped partial: the global chunk index plus the deterministic `k·k`
/// per-chunk Gram. This is the only message that crosses the node boundary —
/// `k·k` f64 values per chunk, never rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodePartial {
    /// Rank that produced this partial (its assignment is validated on
    /// receipt, so a misconfigured worker is rejected loudly).
    pub rank: usize,
    /// Global chunk index of the partial.
    pub chunk_index: usize,
    /// Flattened `k·k` row-major per-chunk Gram, as produced by
    /// [`chunk_gram_flat`] over the chunk's rows.
    pub gram: Vec<f64>,
}

/// Serialized cursor of one worker: everything needed for a **replacement**
/// process (same rank, any host) to continue the dead worker's deterministic
/// chunk sequence from where receipts stopped. Pure data; the worker's row
/// source re-seeks by row range, which is a pure function of the partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeWorkerCheckpoint {
    pub partition: CrossNodePartition,
    pub rank: usize,
    /// Ordinal (into the rank's owned-chunk sequence) of the next chunk to
    /// compute and ship. Everything below it has been durably shipped.
    pub next_ordinal: usize,
}

/// Worker-side driver for one rank: walks the rank's deterministic owned-chunk
/// sequence, turning row slices into shippable [`NodePartial`]s.
///
/// The worker does **not** do I/O: the caller streams rows (from its shards /
/// object store) for the row range [`NodeWorker::next_chunk_rows`] names, hands
/// them to [`NodeWorker::emit`], and ships the returned partial. The cursor
/// advances only on `emit`, so "ship durably, then checkpoint" gives exactly-
/// once production under crash-resume (re-shipping an already-folded chunk is
/// rejected by the coordinator as a duplicate, which is the safe failure).
#[derive(Clone, Debug)]
pub struct NodeWorker {
    partition: CrossNodePartition,
    rank: usize,
    next_ordinal: usize,
}

impl NodeWorker {
    /// Fresh worker for `rank`, starting at the beginning of its sequence.
    pub fn new(partition: CrossNodePartition, rank: usize) -> Result<Self, String> {
        if rank >= partition.n_ranks {
            return Err(format!(
                "NodeWorker: rank {rank} out of range (n_ranks = {})",
                partition.n_ranks
            ));
        }
        Ok(Self {
            partition,
            rank,
            next_ordinal: 0,
        })
    }

    /// Resume a (replacement) worker from a serialized cursor. Validates the
    /// cursor against the partition so a checkpoint from a different pass is
    /// rejected loudly.
    pub fn resume(state: NodeWorkerCheckpoint) -> Result<Self, String> {
        if state.rank >= state.partition.n_ranks {
            return Err(format!(
                "NodeWorkerCheckpoint: rank {} out of range (n_ranks = {})",
                state.rank, state.partition.n_ranks
            ));
        }
        let owned = state.partition.chunks_owned_by(state.rank);
        if state.next_ordinal > owned {
            return Err(format!(
                "NodeWorkerCheckpoint: next_ordinal {} exceeds owned chunk count {owned}",
                state.next_ordinal
            ));
        }
        Ok(Self {
            partition: state.partition,
            rank: state.rank,
            next_ordinal: state.next_ordinal,
        })
    }

    /// Serialize the cursor. Write this (durably) after each successful ship.
    pub fn checkpoint(&self) -> NodeWorkerCheckpoint {
        NodeWorkerCheckpoint {
            partition: self.partition,
            rank: self.rank,
            next_ordinal: self.next_ordinal,
        }
    }

    /// `true` once this rank's sequence is exhausted.
    pub fn is_done(&self) -> bool {
        self.partition
            .owned_chunk(self.rank, self.next_ordinal)
            .is_none()
    }

    /// Global chunk index and row range of the next chunk to compute, or
    /// `None` when done. The caller fetches exactly these rows.
    pub fn next_chunk_rows(&self) -> Option<(usize, std::ops::Range<usize>)> {
        let idx = self.partition.owned_chunk(self.rank, self.next_ordinal)?;
        Some((idx, self.partition.chunk_rows(idx)))
    }

    /// Compute the next chunk's deterministic partial from its rows and advance
    /// the cursor. `rows` must be exactly the rows of
    /// [`NodeWorker::next_chunk_rows`] (shape-validated here; content is the
    /// caller's contract, same as the in-process path).
    pub fn emit(&mut self, rows: ArrayView2<'_, f64>) -> Result<NodePartial, String> {
        let (chunk_index, range) = self
            .next_chunk_rows()
            .ok_or_else(|| format!("NodeWorker rank {}: sequence exhausted", self.rank))?;
        if rows.nrows() != range.len() || rows.ncols() != self.partition.border_dim {
            return Err(format!(
                "NodeWorker rank {}: chunk {chunk_index} has shape ({}, {}) but expected ({}, {})",
                self.rank,
                rows.nrows(),
                rows.ncols(),
                range.len(),
                self.partition.border_dim
            ));
        }
        let gram = chunk_gram_flat(rows);
        self.next_ordinal += 1;
        Ok(NodePartial {
            rank: self.rank,
            chunk_index,
            gram,
        })
    }
}

/// Serializable coordinator state: the inner accumulation state plus the
/// per-rank receipt cursors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrossNodeCheckpoint {
    pub partition: CrossNodePartition,
    /// The wrapped [`StreamingBorderGram`] state (fold forest + pending +
    /// chunk frontier).
    pub inner: BorderGramCheckpoint,
    /// Per-rank count of partials received so far — each is an ordinal cursor
    /// into that rank's deterministic owned-chunk sequence, used both for
    /// receipt validation (in-sequence, no gaps per rank) and to tell a
    /// restarted fleet where each rank should resume.
    pub received_per_rank: Vec<usize>,
}

/// Coordinator-side reduction: receives [`NodePartial`]s from the fleet and
/// folds them into the single global fixed-tree accumulator.
///
/// Receipt validation is per rank and in-sequence: rank `r`'s `i`-th accepted
/// partial must be its `i`-th owned chunk. This makes the per-rank cursor in
/// [`CrossNodeCheckpoint::received_per_rank`] a complete description of what
/// has been received, which is what lets a dead rank resume from a bare
/// ordinal. Cross-rank arrival order is unconstrained (the inner accumulator
/// buffers out-of-order chunks), so slow nodes never block fast ones.
pub struct CrossNodeGramReduction {
    partition: CrossNodePartition,
    inner: StreamingBorderGram,
    received_per_rank: Vec<usize>,
}

impl CrossNodeGramReduction {
    /// Fresh coordinator for the given partition.
    pub fn new(partition: CrossNodePartition) -> Result<Self, String> {
        let inner =
            StreamingBorderGram::new(partition.border_dim, partition.n_rows, partition.chunk_size)?;
        Ok(Self {
            received_per_rank: vec![0; partition.n_ranks],
            partition,
            inner,
        })
    }

    /// The shared partition (workers must be constructed with an equal one).
    pub fn partition(&self) -> CrossNodePartition {
        self.partition
    }

    /// How many partials rank `rank` has had accepted — the ordinal a
    /// replacement worker for that rank should resume from.
    pub fn rank_cursor(&self, rank: usize) -> Option<usize> {
        self.received_per_rank.get(rank).copied()
    }

    /// `true` once every chunk of every rank has been received and folded.
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Receive one shipped partial. Validates rank, ownership, and per-rank
    /// sequence position, then folds through
    /// [`StreamingBorderGram::submit_chunk_gram`] (which re-validates index
    /// range, duplicates, and partial shape). A duplicate of an already-folded
    /// chunk — the signature of an at-least-once transport retry or a worker
    /// that resumed from a stale cursor — is rejected with an error naming the
    /// chunk, never silently double-counted.
    pub fn receive(&mut self, partial: NodePartial) -> Result<(), String> {
        let NodePartial {
            rank,
            chunk_index,
            gram,
        } = partial;
        if rank >= self.partition.n_ranks {
            return Err(format!(
                "CrossNodeGramReduction: rank {rank} out of range (n_ranks = {})",
                self.partition.n_ranks
            ));
        }
        if self.partition.owner_rank(chunk_index) != rank {
            return Err(format!(
                "CrossNodeGramReduction: chunk {chunk_index} is owned by rank {}, not rank {rank}",
                self.partition.owner_rank(chunk_index)
            ));
        }
        let cursor = self.received_per_rank[rank];
        match self.partition.owned_chunk(rank, cursor) {
            Some(expected) if expected == chunk_index => {}
            Some(expected) => {
                return Err(format!(
                    "CrossNodeGramReduction: rank {rank} shipped chunk {chunk_index} but its \
                     cursor expects chunk {expected} (ordinal {cursor}); a worker resumed from \
                     a stale or future checkpoint"
                ));
            }
            None => {
                return Err(format!(
                    "CrossNodeGramReduction: rank {rank} shipped chunk {chunk_index} past the \
                     end of its owned sequence"
                ));
            }
        }
        self.inner.submit_chunk_gram(chunk_index, gram)?;
        self.received_per_rank[rank] = cursor + 1;
        Ok(())
    }

    /// Serialize the full coordinator state. Resume-equals-straight-through is
    /// inherited bit-for-bit from the inner accumulator; the per-rank cursors
    /// resume receipt validation exactly where it stopped.
    pub fn checkpoint(&self) -> CrossNodeCheckpoint {
        CrossNodeCheckpoint {
            partition: self.partition,
            inner: self.inner.checkpoint(),
            received_per_rank: self.received_per_rank.clone(),
        }
    }

    /// Reconstruct a coordinator from a checkpoint, validating the cursor
    /// structure against the partition so corruption is rejected loudly.
    pub fn resume(state: CrossNodeCheckpoint) -> Result<Self, String> {
        if state.received_per_rank.len() != state.partition.n_ranks {
            return Err(format!(
                "CrossNodeCheckpoint: {} rank cursors for n_ranks = {}",
                state.received_per_rank.len(),
                state.partition.n_ranks
            ));
        }
        if state.inner.border_dim != state.partition.border_dim
            || state.inner.n_rows != state.partition.n_rows
            || state.inner.chunk_size != state.partition.chunk_size
        {
            return Err(
                "CrossNodeCheckpoint: inner accumulator partition disagrees with the cross-node \
                 partition"
                    .to_string(),
            );
        }
        for (rank, &cursor) in state.received_per_rank.iter().enumerate() {
            if cursor > state.partition.chunks_owned_by(rank) {
                return Err(format!(
                    "CrossNodeCheckpoint: rank {rank} cursor {cursor} exceeds its owned chunk \
                     count {}",
                    state.partition.chunks_owned_by(rank)
                ));
            }
        }
        let inner = StreamingBorderGram::resume(state.inner)?;
        Ok(Self {
            partition: state.partition,
            inner,
            received_per_rank: state.received_per_rank,
        })
    }

    /// Finish the pass, returning the `k×k` border Gram. Errors if any rank's
    /// sequence is incomplete. The result is a pure function of the row content
    /// and `(n_rows, chunk_size)` — identical bits for any node count, any
    /// arrival interleaving, and any checkpoint/resume history on either side.
    pub fn finish(self) -> Result<Array2<f64>, String> {
        for (rank, &cursor) in self.received_per_rank.iter().enumerate() {
            let owned = self.partition.chunks_owned_by(rank);
            if cursor != owned {
                return Err(format!(
                    "CrossNodeGramReduction: finish() with rank {rank} at ordinal {cursor} of \
                     {owned} owned chunks"
                ));
            }
        }
        self.inner.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::s;

    /// Deterministic pseudo-random row matrix keyed purely by index (same
    /// recipe as the streaming_border tests, so cross-file comparisons hold).
    fn planted_rows(n: usize, k: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, k), |(i, j)| {
            let x = (i as f64 + 1.0) * 0.7390851 + (j as f64 + 1.0) * 1.6180339;
            (x.sin() * 43_758.547).fract() * 2.0 - 1.0
        })
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

    /// Run a whole fleet of `n_ranks` workers to completion against one
    /// coordinator, interleaving ranks round-robin with a deterministic skew so
    /// arrival order exercises the pending buffer. Returns the final Gram.
    fn run_fleet(rows: &Array2<f64>, chunk_size: usize, n_ranks: usize) -> Array2<f64> {
        let partition =
            CrossNodePartition::new(rows.ncols(), rows.nrows(), chunk_size, n_ranks).unwrap();
        let mut coordinator = CrossNodeGramReduction::new(partition).unwrap();
        let mut workers: Vec<NodeWorker> = (0..n_ranks)
            .map(|r| NodeWorker::new(partition, r).unwrap())
            .collect();
        // Deterministic skewed interleaving: each sweep lets rank r ship
        // (r % 3 + 1) chunks, so ranks run ahead/behind each other and the
        // coordinator's out-of-order pending path is exercised.
        let mut any_live = true;
        while any_live {
            any_live = false;
            for (r, worker) in workers.iter_mut().enumerate() {
                for _ in 0..(r % 3 + 1) {
                    let Some((_, range)) = worker.next_chunk_rows() else {
                        break;
                    };
                    let partial = worker.emit(rows.slice(s![range, ..])).unwrap();
                    coordinator.receive(partial).unwrap();
                    any_live = true;
                }
                if !worker.is_done() {
                    any_live = true;
                }
            }
        }
        assert!(coordinator.is_complete());
        coordinator.finish().unwrap()
    }

    #[test]
    fn node_count_never_changes_bits() {
        // The frontier invariant: 1, 3, and 5 nodes produce the identical bit
        // pattern, and all match the single-process StreamingBorderGram.
        let n = 977; // not a multiple of the chunk size
        let k = 4;
        let chunk_size = 7;
        let rows = planted_rows(n, k);

        let mut single = StreamingBorderGram::new(k, n, chunk_size).unwrap();
        for j in 0..single.n_chunks() {
            let range = single.chunk_rows(j);
            single.submit_chunk(j, rows.slice(s![range, ..])).unwrap();
        }
        let reference = single.finish().unwrap();

        for n_ranks in [1usize, 3, 5] {
            let fleet = run_fleet(&rows, chunk_size, n_ranks);
            assert_bit_identical(
                &reference,
                &fleet,
                &format!("single-process vs {n_ranks}-node fleet"),
            );
        }
    }

    #[test]
    fn dead_node_resumes_from_cursor_bit_identically() {
        let n = 530;
        let k = 3;
        let chunk_size = 5;
        let n_ranks = 3;
        let rows = planted_rows(n, k);
        let reference = run_fleet(&rows, chunk_size, n_ranks);

        let partition = CrossNodePartition::new(k, n, chunk_size, n_ranks).unwrap();
        let mut coordinator = CrossNodeGramReduction::new(partition).unwrap();
        let mut workers: Vec<NodeWorker> = (0..n_ranks)
            .map(|r| NodeWorker::new(partition, r).unwrap())
            .collect();

        // Rank 1 ships 4 chunks, checkpoints durably, then "dies". The other
        // ranks ship a few chunks too.
        let mut rank1_cursor = None;
        for (r, worker) in workers.iter_mut().enumerate() {
            let ship = if r == 1 { 4 } else { 2 };
            for _ in 0..ship {
                let Some((_, range)) = worker.next_chunk_rows() else {
                    break;
                };
                let partial = worker.emit(rows.slice(s![range, ..])).unwrap();
                coordinator.receive(partial).unwrap();
            }
            if r == 1 {
                let json = serde_json::to_string(&worker.checkpoint()).unwrap();
                rank1_cursor = Some(json);
            }
        }
        workers.remove(1); // the death: rank-1 worker removed and dropped here

        // Coordinator also survives a checkpoint round-trip mid-pass.
        let coord_json = serde_json::to_string(&coordinator.checkpoint()).unwrap();
        drop(coordinator);
        let restored: CrossNodeCheckpoint = serde_json::from_str(&coord_json).unwrap();
        let mut coordinator = CrossNodeGramReduction::resume(restored).unwrap();

        // A replacement process resumes rank 1 from its serialized cursor; the
        // coordinator's own cursor agrees with it.
        let cp: NodeWorkerCheckpoint = serde_json::from_str(&rank1_cursor.unwrap()).unwrap();
        assert_eq!(coordinator.rank_cursor(1), Some(cp.next_ordinal));
        let replacement = NodeWorker::resume(cp).unwrap();
        workers.insert(1, replacement);

        // Drain the fleet to completion.
        let mut any_live = true;
        while any_live {
            any_live = false;
            for worker in workers.iter_mut() {
                if let Some((_, range)) = worker.next_chunk_rows() {
                    let partial = worker.emit(rows.slice(s![range, ..])).unwrap();
                    coordinator.receive(partial).unwrap();
                    any_live = true;
                }
            }
        }
        let resumed = coordinator.finish().unwrap();
        assert_bit_identical(&reference, &resumed, "death-resume vs straight-through");
    }

    #[test]
    fn receipt_validation_rejects_misrouted_and_out_of_sequence_partials() {
        let n = 60;
        let k = 2;
        let chunk_size = 4; // 15 chunks
        let n_ranks = 3;
        let rows = planted_rows(n, k);
        let partition = CrossNodePartition::new(k, n, chunk_size, n_ranks).unwrap();
        let mut coordinator = CrossNodeGramReduction::new(partition).unwrap();
        let mut w0 = NodeWorker::new(partition, 0).unwrap();

        let (idx, range) = w0.next_chunk_rows().unwrap();
        assert_eq!(idx, 0);
        let mut partial = w0.emit(rows.slice(s![range, ..])).unwrap();

        // Misrouted: claim the partial came from rank 1 (which does not own
        // chunk 0).
        partial.rank = 1;
        let err = coordinator.receive(partial.clone()).unwrap_err();
        assert!(err.contains("owned by rank 0"), "got: {err}");

        // Correctly routed: accepted.
        partial.rank = 0;
        coordinator.receive(partial.clone()).unwrap();

        // Duplicate (transport retry): rejected, never double-counted.
        let err = coordinator.receive(partial).unwrap_err();
        assert!(err.contains("cursor expects chunk 3"), "got: {err}");

        // Out of sequence: rank 0's cursor expects its ordinal-1 chunk (global
        // chunk 3), not its ordinal-2 chunk (global chunk 6).
        let (idx, range) = w0.next_chunk_rows().unwrap();
        assert_eq!(idx, 3);
        let skipped = w0.emit(rows.slice(s![range, ..])).unwrap();
        let (idx6, range6) = w0.next_chunk_rows().unwrap();
        assert_eq!(idx6, 6);
        let ahead = w0.emit(rows.slice(s![range6, ..])).unwrap();
        let err = coordinator.receive(ahead).unwrap_err();
        assert!(err.contains("expects chunk 3"), "got: {err}");
        coordinator.receive(skipped).unwrap();
    }

    #[test]
    fn assignment_is_a_pure_partition() {
        // Every chunk is owned by exactly one rank, and the per-rank owned
        // sequences tile [0, n_chunks) — for several fleet sizes.
        for (n_rows, chunk_size, n_ranks) in [(100, 7, 1), (100, 7, 4), (3, 10, 8), (0, 5, 3)] {
            let partition = CrossNodePartition::new(2, n_rows, chunk_size, n_ranks).unwrap();
            let n_chunks = partition.n_chunks();
            let mut seen = vec![false; n_chunks];
            let mut total = 0usize;
            for rank in 0..n_ranks {
                let owned = partition.chunks_owned_by(rank);
                for ordinal in 0..owned {
                    let idx = partition.owned_chunk(rank, ordinal).unwrap();
                    assert_eq!(partition.owner_rank(idx), rank);
                    assert!(!seen[idx], "chunk {idx} assigned twice");
                    seen[idx] = true;
                    total += 1;
                }
                assert!(partition.owned_chunk(rank, owned).is_none());
            }
            assert_eq!(total, n_chunks, "assignment must tile all chunks");
        }
    }
}
