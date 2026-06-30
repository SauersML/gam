//! Tiled scoring and online top-`s` selection.
//!
//! The router must rank every row against all `K` atoms without ever holding an
//! `N×K` score matrix. We do this by GEMM-ing the minibatch against the
//! dictionary one **column tile** at a time (`atoms_tile` of shape `tile × P`),
//! producing a `rows × tile` score block, folding that block into a per-row
//! online top-`s` selector, and discarding it. Peak score memory is therefore
//! `rows × tile`, independent of `K`.

use ndarray::{ArrayView1, ArrayView2, Axis};

/// Online "keep the `s` largest-magnitude scores seen so far" selector for a
/// single row. Selection is by `|score|` (the dictionary atoms are unit-norm,
/// so `|xᵀd|` is the magnitude of the optimal 1-atom projection); ties break by
/// smaller atom index for determinism.
#[derive(Clone, Debug)]
pub struct TopSSelector {
    /// `(atom_index, score, |score|)`, length ≤ `s`, kept unsorted.
    heap: Vec<(u32, f32, f32)>,
    capacity: usize,
}

impl TopSSelector {
    pub fn new(capacity: usize) -> Self {
        Self {
            heap: Vec::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
        }
    }

    /// Offer one `(atom, score)` candidate.
    #[inline]
    pub fn offer(&mut self, atom: u32, score: f32) {
        let mag = score.abs();
        if self.heap.len() < self.capacity {
            self.heap.push((atom, score, mag));
            return;
        }
        // Find the current weakest slot.
        let mut worst = 0usize;
        for k in 1..self.heap.len() {
            if self.heap[k].2 < self.heap[worst].2
                || (self.heap[k].2 == self.heap[worst].2 && self.heap[k].0 > self.heap[worst].0)
            {
                worst = k;
            }
        }
        let (w_atom, _, w_mag) = self.heap[worst];
        if mag > w_mag || (mag == w_mag && atom < w_atom) {
            self.heap[worst] = (atom, score, mag);
        }
    }

    /// Finalise, returning `(atom, score)` pairs sorted by descending `|score|`
    /// (ties by ascending atom index).
    pub fn finish(mut self) -> Vec<(u32, f32)> {
        self.heap.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        self.heap.into_iter().map(|(a, s, _)| (a, s)).collect()
    }
}

/// Score a row against a tile of atoms (`atoms_tile`, `tile × P`, rows are
/// atoms) and fold every score into `sel`. `atom_offset` is the global index of
/// the tile's first atom.
#[inline]
pub fn score_row_tile(
    row: ArrayView1<'_, f32>,
    atoms_tile: ArrayView2<'_, f32>,
    atom_offset: usize,
    sel: &mut TopSSelector,
) {
    let p = row.len();
    for (local, atom) in atoms_tile.outer_iter().enumerate() {
        let mut acc = 0.0f32;
        for c in 0..p {
            acc += row[c] * atom[c];
        }
        sel.offer((atom_offset + local) as u32, acc);
    }
}

/// Convenience: full top-`s` selection of one row against the entire decoder,
/// tiled internally. Returns `(atom, score)` pairs, ≤ `s` of them, sorted by
/// descending `|score|`. Used by tests and by the router's per-row path.
pub fn top_s_online(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    s: usize,
    tile: usize,
) -> Vec<(u32, f32)> {
    let k = decoder.nrows();
    let tile = tile.max(1);
    let mut sel = TopSSelector::new(s);
    let mut start = 0usize;
    while start < k {
        let end = (start + tile).min(k);
        let block = decoder.slice(ndarray::s![start..end, ..]);
        score_row_tile(row, block, start, &mut sel);
        start = end;
    }
    sel.finish()
}

/// A reusable tiled scorer over a fixed decoder. Holds the tile width so the
/// router can score a whole minibatch with one object, never materialising an
/// `N×K` block.
#[derive(Clone, Copy, Debug)]
pub struct TileScorer {
    pub tile: usize,
    pub active: usize,
}

impl TileScorer {
    pub fn new(active: usize, tile: usize) -> Self {
        Self {
            tile: tile.max(1),
            active: active.max(1),
        }
    }

    /// Top-`active` atoms for `row` against `decoder`.
    pub fn route_row(
        &self,
        row: ArrayView1<'_, f32>,
        decoder: ArrayView2<'_, f32>,
    ) -> Vec<(u32, f32)> {
        top_s_online(row, decoder, self.active, self.tile)
    }

    /// Top-`active` atoms for every row of a minibatch `rows` (`B × P`) against
    /// `decoder` (`K × P`), scored a column tile at a time via a batched GEMM.
    ///
    /// This is the implementation that delivers the module's promise: the score
    /// block formed at any instant is `B × tile` (peak `rows × tile`,
    /// independent of `K`), and each tile is a single `(B × P)·(P × tile)`
    /// matrix multiply rather than `B × tile` scalar dot loops. The online
    /// top-`s` selector sees the atoms in the same global order as
    /// [`Self::route_row`] (tile 0 first, ascending atom index), so the
    /// shortlist agrees with the row-at-a-time path; the GEMM contracts the same
    /// `P` terms but `matrixmultiply` may accumulate them in a blocked order, so
    /// scores agree to f32 rounding rather than bit-for-bit. Returns one
    /// `(atom, score)` shortlist per row, in row order.
    pub fn route_minibatch(
        &self,
        rows: ArrayView2<'_, f32>,
        decoder: ArrayView2<'_, f32>,
    ) -> Vec<Vec<(u32, f32)>> {
        let b = rows.nrows();
        let k = decoder.nrows();
        let mut selectors: Vec<TopSSelector> =
            (0..b).map(|_| TopSSelector::new(self.active)).collect();

        let mut start = 0usize;
        while start < k {
            let end = (start + self.tile).min(k);
            // `decoder` tile is `tile × P`; transpose to `P × tile` so the GEMM
            // produces the `B × tile` score block directly (rows × atoms).
            let tile_block = decoder.slice(ndarray::s![start..end, ..]);
            let scores = rows.dot(&tile_block.t()); // (B × P)·(P × tile) = B × tile
            for (local, score_col) in scores.axis_iter(Axis(1)).enumerate() {
                let atom = (start + local) as u32;
                for (row_idx, &sc) in score_col.iter().enumerate() {
                    selectors[row_idx].offer(atom, sc);
                }
            }
            start = end;
        }
        selectors.into_iter().map(TopSSelector::finish).collect()
    }
}
