//! Tiled scoring and online top-`s` selection.
//!
//! The router must rank every row against all `K` atoms without ever holding an
//! `N×K` score matrix. We do this by GEMM-ing the minibatch against the
//! dictionary one **column tile** at a time (`atoms_tile` of shape `tile × P`),
//! producing a `rows × tile` score block, folding that block into a per-row
//! online top-`s` selector, and discarding it. Peak score memory is therefore
//! `rows × tile`, independent of `K`.

use ndarray::{ArrayView1, ArrayView2, Axis};

/// Minibatch score route path selected by [`TileScorer::route_minibatch_with_mode`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreRoutePath {
    /// The CUDA score-block router computed the minibatch scores.
    Device,
    /// A host router computed the minibatch scores.
    Cpu,
}

/// Routed minibatch plus the dispatch plan/path used to produce it.
#[derive(Clone, Debug)]
pub struct ScoreRouteResult {
    /// One top-`active` shortlist per input row.
    pub selections: Vec<Vec<(u32, f32)>>,
    /// CPU/device path that produced the score route.
    pub path: ScoreRoutePath,
    /// Shape, admission, and tile geometry for this score route.
    pub plan: gam_gpu::DictionaryScoreRoutePlan,
    /// Bytes copied device→host by the route itself. The fused CUDA route copies
    /// only `(atom, score)` shortlists (`rows × active × 8` bytes), never the
    /// full `rows × K` score stream.
    pub device_dtoh_bytes: usize,
    /// Bytes an unfused/full-score CUDA route would have copied device→host for
    /// this minibatch by returning every score tile (`rows × K × sizeof(f32)`).
    pub unfused_score_dtoh_bytes: usize,
}

/// Aggregate score-route counters for a sparse-dictionary fit or stream.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScoreRouteStats {
    /// Minibatches routed.
    pub minibatches: usize,
    /// Minibatches whose shape cleared the device admission floor.
    pub admitted_minibatches: usize,
    /// Minibatches scored by the CUDA router.
    pub device_minibatches: usize,
    /// Minibatches scored on the host.
    pub cpu_minibatches: usize,
    /// Total route score elements (`rows * atoms`) considered.
    pub score_elements: u128,
    /// Total number of score tiles walked across all minibatches.
    pub score_tiles: usize,
    /// Maximum score-block bytes held by any single route tile.
    pub peak_score_bytes: usize,
    /// Lower-bound dot-product arithmetic across all routed scores.
    pub dot_flops_lower_bound: u128,
    /// Actual route bytes copied device→host. For the fused CUDA route this is
    /// shortlist-sized (`rows × active × 8`) rather than score-matrix-sized.
    pub device_dtoh_bytes: u128,
    /// Device→host score bytes avoided by the fused CUDA route, relative to a
    /// route that downloads every `rows × tile` score block.
    pub unfused_score_dtoh_bytes_avoided: u128,
}

impl ScoreRouteStats {
    /// Fold one routed minibatch into the aggregate counters.
    pub fn record(&mut self, plan: gam_gpu::DictionaryScoreRoutePlan, path: ScoreRoutePath) {
        self.record_with_device_transfer(plan, path, 0, 0);
    }

    /// Fold a full route result into the aggregate counters, including its
    /// measured device→host transfer accounting.
    pub fn record_result(&mut self, result: &ScoreRouteResult) {
        self.record_with_device_transfer(
            result.plan,
            result.path,
            result.device_dtoh_bytes,
            result.unfused_score_dtoh_bytes,
        );
    }

    fn record_with_device_transfer(
        &mut self,
        plan: gam_gpu::DictionaryScoreRoutePlan,
        path: ScoreRoutePath,
        device_dtoh_bytes: usize,
        unfused_score_dtoh_bytes: usize,
    ) {
        self.minibatches += 1;
        if plan.device_admitted {
            self.admitted_minibatches += 1;
        }
        match path {
            ScoreRoutePath::Device => self.device_minibatches += 1,
            ScoreRoutePath::Cpu => self.cpu_minibatches += 1,
        }
        self.score_elements = self
            .score_elements
            .saturating_add((plan.n_rows as u128).saturating_mul(plan.n_items as u128));
        self.score_tiles = self.score_tiles.saturating_add(plan.tile_count);
        self.peak_score_bytes = self.peak_score_bytes.max(plan.peak_score_bytes);
        self.dot_flops_lower_bound = self
            .dot_flops_lower_bound
            .saturating_add(plan.dot_flops_lower_bound);
        self.device_dtoh_bytes = self
            .device_dtoh_bytes
            .saturating_add(device_dtoh_bytes as u128);
        self.unfused_score_dtoh_bytes_avoided =
            self.unfused_score_dtoh_bytes_avoided.saturating_add(
                (unfused_score_dtoh_bytes as u128).saturating_sub(device_dtoh_bytes as u128),
            );
    }

    /// Merge another aggregate into this one.
    pub fn absorb(&mut self, other: Self) {
        self.minibatches = self.minibatches.saturating_add(other.minibatches);
        self.admitted_minibatches = self
            .admitted_minibatches
            .saturating_add(other.admitted_minibatches);
        self.device_minibatches = self
            .device_minibatches
            .saturating_add(other.device_minibatches);
        self.cpu_minibatches = self.cpu_minibatches.saturating_add(other.cpu_minibatches);
        self.score_elements = self.score_elements.saturating_add(other.score_elements);
        self.score_tiles = self.score_tiles.saturating_add(other.score_tiles);
        self.peak_score_bytes = self.peak_score_bytes.max(other.peak_score_bytes);
        self.dot_flops_lower_bound = self
            .dot_flops_lower_bound
            .saturating_add(other.dot_flops_lower_bound);
        self.device_dtoh_bytes = self
            .device_dtoh_bytes
            .saturating_add(other.device_dtoh_bytes);
        self.unfused_score_dtoh_bytes_avoided = self
            .unfused_score_dtoh_bytes_avoided
            .saturating_add(other.unfused_score_dtoh_bytes_avoided);
    }
}

/// Online "keep the `s` largest-magnitude scores seen so far" selector for a
/// single row. Selection is by `|score|` (the dictionary atoms are unit-norm,
/// so `|xᵀd|` is the magnitude of the optimal 1-atom projection); ties break by
/// smaller atom index for determinism.
#[derive(Clone, Debug)]
pub struct TopSSelector {
    /// `(atom_index, score, |score|)`, length ≤ `s`, kept unsorted.
    heap: Vec<(u32, f32, f32)>,
    capacity: usize,
    /// Index of the current weakest slot (smallest `|score|`, ties → larger
    /// atom index). Meaningful only once `heap.len() == capacity`; maintained
    /// incrementally so the common *reject* path is O(1) instead of O(capacity).
    worst_idx: usize,
}

impl TopSSelector {
    pub fn new(capacity: usize) -> Self {
        Self {
            heap: Vec::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            worst_idx: 0,
        }
    }

    /// Rescan the full heap for the weakest slot (smallest `|score|`, ties
    /// broken by *larger* atom index — the slot the next stronger candidate
    /// should evict). Called only when the heap first fills and after each
    /// accepted replacement, never on the (dominant) reject path.
    #[inline]
    fn recompute_worst(&mut self) {
        let mut worst = 0usize;
        for k in 1..self.heap.len() {
            if self.heap[k].2 < self.heap[worst].2
                || (self.heap[k].2 == self.heap[worst].2 && self.heap[k].0 > self.heap[worst].0)
            {
                worst = k;
            }
        }
        self.worst_idx = worst;
    }

    /// Offer one `(atom, score)` candidate.
    ///
    /// Once the heap is full this is O(1) for the overwhelmingly common case of
    /// a candidate that cannot displace the cached weakest slot (at `K ≈ 32_000`
    /// atoms and `s` survivors, all but ~`s` offers are such rejects). Only an
    /// accepted replacement pays the O(capacity) rescan to refresh the cached
    /// weakest. The selection is bit-identical to a fresh full rescan per offer:
    /// the accept test and the weakest-slot definition are unchanged.
    #[inline]
    pub fn offer(&mut self, atom: u32, score: f32) {
        let mag = score.abs();
        if self.heap.len() < self.capacity {
            self.heap.push((atom, score, mag));
            if self.heap.len() == self.capacity {
                self.recompute_worst();
            }
            return;
        }
        // Full: O(1) reject against the cached weakest slot.
        let (w_atom, _, w_mag) = self.heap[self.worst_idx];
        if mag > w_mag || (mag == w_mag && atom < w_atom) {
            self.heap[self.worst_idx] = (atom, score, mag);
            self.recompute_worst();
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
    /// [`Self::route_row`] (tile 0 first, ascending atom index). The GEMM
    /// contracts the same `P` terms but `matrixmultiply` may accumulate them in
    /// a blocked order, so the per-atom scores agree with the row-at-a-time path
    /// only to f32 rounding; where two atoms tie within that rounding the two
    /// paths may select different members of the tie (interchangeable for the
    /// reconstruction, which is why the fit stays minibatch-invariant rather
    /// than bit-identical). Returns one `(atom, score)` shortlist per row, in row
    /// order.
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

    /// Top-`active` atoms for every row, using the sparse-dict CUDA score-block
    /// router when the process-wide GPU mode admits it and the `rows × K` route
    /// clears the device break-even. Auto mode never pays the slower GPU wrapper's
    /// CPU fallback below break-even: it goes straight to the batched CPU GEMM
    /// router above. Required mode is fail-closed and propagates the CUDA router's
    /// error instead of silently routing on the CPU.
    pub fn route_minibatch_dispatch(
        &self,
        rows: ArrayView2<'_, f32>,
        decoder: ArrayView2<'_, f32>,
    ) -> Result<Vec<Vec<(u32, f32)>>, String> {
        Ok(self
            .route_minibatch_with_mode(rows, decoder, gam_gpu::global_policy())?
            .selections)
    }

    /// Top-`active` atoms for every row under an explicit GPU residency mode.
    ///
    /// This is the reusable Rust control surface for high-`K` T1 scoring:
    /// callers can choose `Off`, `Auto`, or fail-closed `Required` per fit/route,
    /// and can inspect the returned [`ScoreRouteResult`] to verify whether a route
    /// was device-admitted and whether it actually used the CUDA score-block path.
    pub fn route_minibatch_with_mode(
        &self,
        rows: ArrayView2<'_, f32>,
        decoder: ArrayView2<'_, f32>,
        mode: gam_gpu::GpuPolicy,
    ) -> Result<ScoreRouteResult, String> {
        let plan = gam_gpu::DictionaryScoreRoutePlan::default_for_shape(
            rows.nrows(),
            decoder.nrows(),
            decoder.ncols(),
        );
        if mode == gam_gpu::GpuPolicy::Off {
            return Ok(ScoreRouteResult {
                selections: self.route_minibatch(rows, decoder),
                path: ScoreRoutePath::Cpu,
                plan,
                device_dtoh_bytes: 0,
                unfused_score_dtoh_bytes: 0,
            });
        }

        if mode == gam_gpu::GpuPolicy::Required && !plan.device_admitted {
            return Err(format!(
                "sparse_dict route_minibatch GpuPolicy::Required: block of {}x{} = {} elems is \
                 below the device launch break-even (DEVICE_SCORE_BLOCK_MIN_ELEMS={}); refusing \
                 to silently run on the CPU",
                plan.n_rows,
                plan.n_items,
                plan.n_rows.saturating_mul(plan.n_items),
                plan.device_min_score_elems
            ));
        }

        #[cfg(target_os = "linux")]
        {
            if mode == gam_gpu::GpuPolicy::Required || plan.device_admitted {
                match super::scoring_gpu::route_minibatch_required(
                    rows,
                    decoder,
                    self.active,
                    self.tile,
                    mode,
                ) {
                    Ok((routed, super::scoring_gpu::ScoreBlockPath::Device, dtoh_bytes)) => {
                        return Ok(ScoreRouteResult {
                            selections: routed,
                            path: ScoreRoutePath::Device,
                            plan,
                            device_dtoh_bytes: dtoh_bytes,
                            unfused_score_dtoh_bytes: plan
                                .n_rows
                                .saturating_mul(plan.n_items)
                                .saturating_mul(std::mem::size_of::<f32>()),
                        });
                    }
                    Ok((routed, super::scoring_gpu::ScoreBlockPath::Cpu, _)) => {
                        if mode == gam_gpu::GpuPolicy::Required {
                            return Err(
                                "sparse_dict route_minibatch Required mode returned CPU path"
                                    .to_string(),
                            );
                        }
                        return Ok(ScoreRouteResult {
                            selections: routed,
                            path: ScoreRoutePath::Cpu,
                            plan,
                            device_dtoh_bytes: 0,
                            unfused_score_dtoh_bytes: 0,
                        });
                    }
                    Err(err) => {
                        if mode == gam_gpu::GpuPolicy::Required {
                            return Err(err.to_string());
                        }
                    }
                }
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            if mode == gam_gpu::GpuPolicy::Required {
                return Err(
                    "sparse_dict route_minibatch GpuPolicy::Required: CUDA routing is only compiled \
                     on Linux"
                        .to_string(),
                );
            }
        }

        Ok(ScoreRouteResult {
            selections: self.route_minibatch(rows, decoder),
            path: ScoreRoutePath::Cpu,
            plan,
            device_dtoh_bytes: 0,
            unfused_score_dtoh_bytes: 0,
        })
    }
}
