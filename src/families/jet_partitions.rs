//! Shared, allocation-free lookup of bitmask set-partitions used by the
//! multi-directional jet `compose_unary` (Faà di Bruno) routines in the
//! marginal-slope and latent-survival families.
//!
//! Each entry `partitions(mask)` is the list of all set-partitions of the bits
//! of `mask`, with each partition represented as a `Vec<usize>` of disjoint
//! sub-masks whose bitwise OR equals `mask`. The total memory footprint is
//! bounded by the Bell numbers up to `B(MAX_DIRS) = 52` for `MAX_DIRS=5`, so
//! the cache is tiny and computed lazily on first use.
use std::sync::OnceLock;

const MAX_DIRS: usize = 5;
const TABLE_LEN: usize = 1usize << MAX_DIRS;

static CACHE: OnceLock<Vec<Vec<Vec<usize>>>> = OnceLock::new();

fn build_partitions(mask: usize) -> Vec<Vec<usize>> {
    if mask == 0 {
        return vec![Vec::new()];
    }
    let first = mask & mask.wrapping_neg();
    let rest = mask ^ first;
    let mut out = Vec::new();
    let mut subset = rest;
    loop {
        let block = first | subset;
        for mut remainder in build_partitions(rest ^ subset) {
            remainder.push(block);
            out.push(remainder);
        }
        if subset == 0 {
            break;
        }
        subset = (subset - 1) & rest;
    }
    out
}

/// Returns the precomputed list of set-partitions for `mask`.
///
/// Supports masks in `0..=(1 << MAX_DIRS) - 1` (i.e. up to `RowKernel<5>`
/// jet machinery). Panics if `mask` is out of range.
pub fn partitions(mask: usize) -> &'static [Vec<usize>] {
    let table = CACHE.get_or_init(|| (0..TABLE_LEN).map(build_partitions).collect());
    &table[mask]
}
