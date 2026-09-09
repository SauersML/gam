//! Bounded reservoir of the worst-reconstructed rows seen in a streaming epoch.
//!
//! Shared by the atom lane (`stream.rs`, dead-atom revival) and the block lane
//! (`block_stream.rs`, dead-block birth proposals): both keep the top-`cap`
//! residual rows by energy with one-shot's deterministic tie-break, and both
//! used to carry a private copy of this type (#2470). The capacity is the
//! caller's: `K` for revival (at most one atom per row, at most `K` dead
//! atoms) and `k_aux · b` for block births. Peak memory is `cap × P` f32 —
//! never `N × K`.

use super::update::DEAD_DENOM;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One candidate row: its residual vector (under the pre-refresh decoder) and
/// the energy used to rank it. Ordered so the [`BinaryHeap`]'s max is the
/// MOST-evictable entry (smallest energy, ties broken toward the larger global
/// index) — that keeps the reservoir holding the worst-reconstructed rows with
/// one-shot's deterministic tie-break (descending energy, ascending row index).
pub(super) struct ResidRow {
    pub(super) norm2: f64,
    pub(super) global_index: u64,
    pub(super) residual: Vec<f32>,
}

impl PartialEq for ResidRow {
    fn eq(&self, other: &Self) -> bool {
        self.norm2 == other.norm2 && self.global_index == other.global_index
    }
}
impl Eq for ResidRow {}
impl Ord for ResidRow {
    fn cmp(&self, other: &Self) -> Ordering {
        // "Greater" == more evictable == smaller residual energy, then larger
        // global index. `total_cmp` keeps this total and NaN-free (norms are
        // finite sums of squares).
        match other.norm2.total_cmp(&self.norm2) {
            Ordering::Equal => self.global_index.cmp(&other.global_index),
            ord => ord,
        }
    }
}
impl PartialOrd for ResidRow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Bounded reservoir of the worst-reconstructed rows seen this epoch.
pub(super) struct ResidualReservoir {
    cap: usize,
    heap: BinaryHeap<ResidRow>,
}

impl ResidualReservoir {
    pub(super) fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            heap: BinaryHeap::new(),
        }
    }

    /// Offer a row's residual to the reservoir. Rows already reconstructed (energy
    /// at or below the dead floor) can seed nothing and are dropped.
    pub(super) fn offer(&mut self, norm2: f64, global_index: u64, residual: Vec<f32>) {
        if norm2 <= DEAD_DENOM {
            return;
        }
        let row = ResidRow {
            norm2,
            global_index,
            residual,
        };
        if self.heap.len() < self.cap {
            self.heap.push(row);
            return;
        }
        // The heap's max is the most-evictable held row; replace it only when the
        // newcomer is strictly LESS evictable (a worse-reconstructed row, or an
        // equal-energy row with a smaller index).
        if let Some(worst_kept) = self.heap.peek() {
            if row.cmp(worst_kept) == Ordering::Less {
                self.heap.pop();
                self.heap.push(row);
            }
        }
    }

    pub(super) fn clear(&mut self) {
        self.heap.clear();
    }

    /// Rows ranked worst-first: descending residual energy, ties by ascending
    /// global index — the one-shot `revive_dead_atoms` /
    /// `dead_block_birth_proposals` order.
    pub(super) fn ranked(&self) -> Vec<&ResidRow> {
        let mut rows: Vec<&ResidRow> = self.heap.iter().collect();
        rows.sort_by(|a, b| {
            b.norm2
                .total_cmp(&a.norm2)
                .then_with(|| a.global_index.cmp(&b.global_index))
        });
        rows
    }
}
