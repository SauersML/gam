//! Partial-fit streaming surface for the block-sparse lane (#1026 block extension).
//!
//! The one-shot [`super::fit_block_sparse_dictionary`] holds the whole `N×P`
//! corpus in memory and alternates route → γ-refresh → frame-refresh → revive over
//! it. For a real corpus (a 30M-row residual-stream harvest) the rows never fit at
//! once, so this module exposes the SAME alternation as a resumable handle a Python
//! loop drives one shard at a time — mirroring [`super::SparseDictStreamState`]:
//!
//! ```text
//! state = BlockSparseStreamState::new(seed, config)     // fit_begin
//! for _epoch in 0..max_epochs {
//!     for shard in shards { state.partial_fit(shard) }   // route + accumulate
//!     state.end_epoch()                                  // γ + frames + revive
//! }
//! state.finalize()                                       // frames + metadata
//! ```
//!
//! All heavy state lives here, native-side: the warm-started block frames, the
//! epoch's accumulated per-block MOD cross-moments (`M_g`, `P×b`), the streaming γ
//! numerator/denominator, per-block usage + within-block code second moments (for
//! the utilisation / stable-rank report), the streaming TSS/RSS moments, and the
//! worst-reconstructed-row reservoir feeding AuxK dead-block revival. A shard
//! round-trips only its own rows through Python — never the `K×P` frames or any
//! `N×K` object — so per-shard overhead is `O(shard × P)`, independent of `K` and
//! of the corpus length.
//!
//! **Equivalence to one-shot.** During an epoch the block frames and the shared
//! scalar γ are FROZEN at their epoch-start values; every shard is routed against
//! them and its cross-moment / γ / moment contributions are summed (all additive),
//! so the assembled `M_g` and (num, den) — and therefore the refreshed frames and
//! γ — are exactly those of a full-batch refresh over the concatenation, up to f32
//! rounding. The polar frame step is the same orthogonal-Procrustes update as the
//! one-shot [`super`] refresh. As with the atom lane, revival residuals are
//! measured under the decoder in force during the pass (the pre-refresh frames),
//! the only deliberate difference from one-shot; the two coincide once the
//! dictionary is populated and revival goes quiescent. Streaming even removes the
//! one-shot loop's slight γ mixing (it uses a single frozen γ for both the code and
//! the reconstruction throughout the epoch), so its per-epoch step is internally
//! consistent.

use super::BlockSparseConfig;
use super::block::{gram_schmidt_rows, route_and_code_all, seed_frames, stable_rank_symmetric};
use crate::frames::GrassmannFrame;
use ndarray::{Array2, ArrayView2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A block that never fired this epoch has self-energy at or below this floor and
/// keeps its current frame (mirrors the one-shot revival's dead threshold).
const DEAD_DENOM: f64 = 1.0e-12;

/// Per-shard summary returned by [`BlockSparseStreamState::partial_fit`].
#[derive(Clone, Copy, Debug)]
pub struct BlockShardStats {
    /// Rows consumed from this shard.
    pub rows: usize,
    /// This shard's reconstruction residual energy `Σ ‖x − x̂‖²` under the frames
    /// in force this epoch (the pre-refresh frames).
    pub rss: f64,
    /// Distinct blocks that have fired at least once so far this epoch (cumulative
    /// across the shards seen since the last [`BlockSparseStreamState::end_epoch`]).
    pub alive_blocks: usize,
}

/// Per-epoch summary returned by [`BlockSparseStreamState::end_epoch`].
#[derive(Clone, Copy, Debug)]
pub struct BlockEpochStats {
    /// Explained variance `1 − RSS/TSS` of the frames routed against this epoch
    /// (the pre-refresh frames), from the streamed TSS/RSS moments.
    pub explained_variance: f64,
    /// Dead blocks revived onto worst-reconstructed residual rows this epoch.
    pub revived: usize,
    /// Dead blocks detected this epoch (fired for no row before revival).
    pub dead: usize,
    /// Refreshed shared tied scalar γ after this epoch.
    pub gamma: f32,
    /// Whether the EV-improvement tolerance was met AND no block was revived.
    pub converged: bool,
    /// Epochs completed so far (this one inclusive).
    pub epoch: usize,
}

/// One candidate row for dead-block revival: its residual vector (under the
/// pre-refresh frames) and the energy used to rank it. Ordered so the
/// [`BinaryHeap`]'s max is the MOST-evictable entry (smallest energy, ties toward
/// the larger global index), keeping the reservoir at the worst-reconstructed rows
/// with the one-shot deterministic tie-break (descending energy, ascending index).
struct ResidRow {
    norm2: f64,
    global_index: u64,
    residual: Vec<f32>,
}

impl PartialEq for ResidRow {
    fn eq(&self, other: &Self) -> bool {
        self.norm2 == other.norm2 && self.global_index == other.global_index
    }
}
impl Eq for ResidRow {}
impl Ord for ResidRow {
    fn cmp(&self, other: &Self) -> Ordering {
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

/// Bounded reservoir of the worst-reconstructed rows seen this epoch. Capacity is
/// `k_aux · b`: revival installs `b` orthonormal rows onto each of at most `k_aux`
/// dead blocks, so the top-`k_aux·b` residual rows are all that can ever be needed.
/// Peak memory is `k_aux·b·P` f32 — never `N×K`.
struct ResidualReservoir {
    cap: usize,
    heap: BinaryHeap<ResidRow>,
}

impl ResidualReservoir {
    fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            heap: BinaryHeap::new(),
        }
    }

    fn offer(&mut self, norm2: f64, global_index: u64, residual: Vec<f32>) {
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
        if let Some(worst_kept) = self.heap.peek() {
            if row.cmp(worst_kept) == Ordering::Less {
                self.heap.pop();
                self.heap.push(row);
            }
        }
    }

    fn clear(&mut self) {
        self.heap.clear();
    }

    /// Rows ranked for revival: descending residual energy, ties by ascending
    /// global index — the one-shot `revive_dead_blocks` order.
    fn ranked(&self) -> Vec<&ResidRow> {
        let mut rows: Vec<&ResidRow> = self.heap.iter().collect();
        rows.sort_by(|a, b| {
            b.norm2
                .total_cmp(&a.norm2)
                .then_with(|| a.global_index.cmp(&b.global_index))
        });
        rows
    }
}

/// Resumable state for a streaming block-sparse fit. Construct with [`Self::new`]
/// (fit_begin), feed shards with [`Self::partial_fit`], close each epoch with
/// [`Self::end_epoch`], and read the frames out with [`Self::finalize`]. The block
/// frames, the shared scalar γ, and the revival state warm-start across every call.
pub struct BlockSparseStreamState {
    config: BlockSparseConfig,
    g: usize,
    b: usize,
    k: usize,
    p: usize,
    decoder: Array2<f32>,
    gamma: f32,

    // ---- accumulators reset at each end_epoch (frozen frames/γ used to fill) ----
    cross: Vec<Array2<f64>>,  // per-block MOD cross-moment M_g (P×b)
    second: Vec<Array2<f64>>, // per-block within-block code 2nd moment (b×b)
    usage: Vec<usize>,
    touched: Vec<bool>,
    alive_count: usize,
    gamma_num: f64,
    gamma_den: f64,
    col_sum: Vec<f64>,
    col_sumsq: Vec<f64>,
    rss: f64,
    row_count: usize,
    reservoir: ResidualReservoir,

    // ---- cross-epoch state ----
    prev_ev: f64,
    last_ev: f64,
    epochs_run: usize,
    last_revived: usize,
    converged: bool,
    last_util: Vec<f32>,
    last_stable: Vec<f32>,
    // Last CLOSED epoch's accumulators, stashed by `end_epoch` before
    // `reset_epoch` zeroes the live ones — the certification read surface
    // (`block_rank_charges`) prices blocks from a COMPLETE epoch, never a
    // partially-filled one.
    last_second: Vec<Array2<f64>>,
    last_usage: Vec<usize>,
    last_rss: f64,
    last_rows: usize,
}

/// Per-block honest-charge ledger over the last closed epoch, as parallel
/// vectors (one entry per block, in block order). The BLOCK is the linear
/// lane's certification unit: its `b` atoms share one jointly-fitted
/// orthonormal frame and one code Gram, so they are priced — and live or
/// die — together. `margin = delta_deviance − charge` in nats;
/// `kept = margin > 0` is the same evidence boundary the hybrid split uses
/// (`Δ(½RSS)/φ̂  vs  ½·d_eff·ln n`, #2124 deviance units).
pub struct BlockRankCharges {
    /// Block index `g` (atom ids are `g*b .. (g+1)*b`).
    pub block: Vec<usize>,
    /// Rows routed to the block over the last closed epoch (`n_eff`).
    pub n_eff: Vec<f64>,
    /// Realised rank-charge DOF of the block's frame under its code Gram.
    pub d_eff: Vec<f64>,
    /// `½·tr(C_g)/φ̂` — the deviance reduction the block's codes claim.
    pub delta_deviance: Vec<f64>,
    /// `½·d_eff·ln n_obs` — the evidence price.
    pub charge: Vec<f64>,
    /// `delta_deviance − charge` (also the Laplace log-evidence-ratio the
    /// e-BH certificate can consume as a `log_e_value`).
    pub margin: Vec<f64>,
    /// `margin > 0`.
    pub kept: Vec<bool>,
}

impl BlockSparseStreamState {
    /// fit_begin: seed the block frames from `seed` (a representative sample) and
    /// prime the epoch accumulators. The seed fixes `P` and the initial
    /// orthonormal frames ([`seed_frames`]); the corpus is streamed later through
    /// [`Self::partial_fit`]. γ starts at 1.
    pub fn new(seed: ArrayView2<'_, f32>, config: &BlockSparseConfig) -> Result<Self, String> {
        validate_config(config)?;
        if seed.nrows() == 0 || seed.ncols() == 0 {
            return Err(
                "BlockSparseStream requires a non-empty seed sample (N×P) to fix P and the initial \
                 block frames"
                    .to_string(),
            );
        }
        if !seed.iter().all(|v| v.is_finite()) {
            return Err("BlockSparseStream seed sample must be finite".to_string());
        }
        let p = seed.ncols();
        if config.block_size > p {
            return Err(format!(
                "BlockSparseStream block_size b={} cannot exceed P={p} (a block's b orthonormal \
                 rows must fit in ℝ^P)",
                config.block_size
            ));
        }
        let g = config.n_blocks;
        let b = config.block_size;
        let k = config.block_topk.min(g).max(1);

        let decoder = seed_frames(seed, g, b);

        let cap = config.aux_k.saturating_mul(b).max(1);
        Ok(Self {
            config: *config,
            g,
            b,
            k,
            p,
            decoder,
            gamma: 1.0,
            cross: (0..g).map(|_| Array2::<f64>::zeros((p, b))).collect(),
            second: (0..g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
            usage: vec![0; g],
            touched: vec![false; g],
            alive_count: 0,
            gamma_num: 0.0,
            gamma_den: 0.0,
            col_sum: vec![0.0; p],
            col_sumsq: vec![0.0; p],
            rss: 0.0,
            row_count: 0,
            reservoir: ResidualReservoir::new(cap),
            prev_ev: f64::NEG_INFINITY,
            last_ev: f64::NEG_INFINITY,
            epochs_run: 0,
            last_revived: 0,
            converged: false,
            last_util: vec![0.0; g],
            last_stable: vec![0.0; g],
            last_second: (0..g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
            last_usage: vec![0; g],
            last_rss: 0.0,
            last_rows: 0,
        })
    }

    /// partial_fit: route + tied-code one shard against the FROZEN epoch frames/γ
    /// and fold its contributions into this epoch's accumulators. Reuses the exact
    /// block-tiled router/coder of the one-shot lane ([`route_and_code_all`]), so
    /// streaming the shards yields the same accumulated cross-moments / γ system as
    /// one full-batch pass over the concatenation.
    pub fn partial_fit(&mut self, shard: ArrayView2<'_, f32>) -> Result<BlockShardStats, String> {
        if shard.nrows() == 0 {
            return Ok(BlockShardStats {
                rows: 0,
                rss: 0.0,
                alive_blocks: self.alive_count,
            });
        }
        if shard.ncols() != self.p {
            return Err(format!(
                "BlockSparseStream.partial_fit: shard has P={} columns but the fit was begun with \
                 P={}",
                shard.ncols(),
                self.p
            ));
        }
        if !shard.iter().all(|v| v.is_finite()) {
            return Err("BlockSparseStream.partial_fit shard must be finite".to_string());
        }

        let p = self.p;
        let b = self.b;
        let gamma = self.gamma;
        let aux_on = self.config.aux_k > 0;
        let codes = route_and_code_all(
            shard,
            self.decoder.view(),
            gamma,
            self.g,
            b,
            self.k,
            self.config.minibatch,
            self.config.block_tile,
        );

        let base_index = self.row_count as u64;
        let mut shard_rss = 0.0f64;
        for (r, code) in codes.iter().enumerate() {
            let xi = shard.row(r);
            for c in 0..p {
                let v = xi[c] as f64;
                self.col_sum[c] += v;
                self.col_sumsq[c] += v * v;
            }

            // Per selected block: its within-block code z (b) and its γ-free
            // subspace contribution proj = Σ_r w_r D_g[r] (P). Accumulate x̂ = γ·Σ proj
            // and the γ-free projection sum p_i = Σ proj (for the γ least-squares).
            let mut sel: Vec<(usize, Vec<f32>, Vec<f32>)> = Vec::with_capacity(self.k);
            let mut xhat = vec![0.0f32; p];
            let mut proj_sum = vec![0.0f32; p];
            for j in 0..code.blocks.len() {
                if code.gates[j] == 0.0 {
                    continue;
                }
                let gg = code.blocks[j] as usize;
                let mut w = vec![0.0f32; b];
                for (rr, wr) in w.iter_mut().enumerate() {
                    let atom = self.decoder.row(gg * b + rr);
                    let mut acc = 0.0f32;
                    for (xc, ac) in xi.iter().zip(atom.iter()) {
                        acc += *xc * *ac;
                    }
                    *wr = acc;
                }
                let mut proj = vec![0.0f32; p];
                for (rr, &wr) in w.iter().enumerate() {
                    if wr == 0.0 {
                        continue;
                    }
                    let atom = self.decoder.row(gg * b + rr);
                    for c in 0..p {
                        proj[c] += wr * atom[c];
                    }
                }
                for c in 0..p {
                    xhat[c] += gamma * proj[c];
                    proj_sum[c] += proj[c];
                }
                let z: Vec<f32> = w.iter().map(|v| gamma * v).collect();
                sel.push((gg, z, proj));
            }

            // Full residual under the frozen model + streaming RSS/reservoir.
            let mut residual = vec![0.0f32; p];
            let mut norm2 = 0.0f64;
            for c in 0..p {
                residual[c] = xi[c] - xhat[c];
                norm2 += residual[c] as f64 * residual[c] as f64;
            }
            shard_rss += norm2;
            if aux_on {
                self.reservoir
                    .offer(norm2, base_index + r as u64, residual.clone());
            }

            // Streaming γ least-squares: γ* = (Σ⟨x,p⟩)/(Σ‖p‖²).
            for c in 0..p {
                self.gamma_num += xi[c] as f64 * proj_sum[c] as f64;
                self.gamma_den += proj_sum[c] as f64 * proj_sum[c] as f64;
            }

            // Per-block MOD cross-moment M_g += r_ig ⊗ z, with the leave-one-block-
            // out residual r_ig = x − x̂ + decode_g = residual + γ·proj_g; plus the
            // within-block code 2nd moment for the stable-rank report.
            for (gg, z, proj) in sel.iter() {
                let gg = *gg;
                if !self.touched[gg] {
                    self.touched[gg] = true;
                    self.alive_count += 1;
                }
                self.usage[gg] += 1;
                let mg = &mut self.cross[gg];
                for c in 0..p {
                    let r_ig_c = residual[c] as f64 + (gamma * proj[c]) as f64;
                    for (rr, &zr) in z.iter().enumerate() {
                        mg[[c, rr]] += r_ig_c * zr as f64;
                    }
                }
                let sg = &mut self.second[gg];
                for r1 in 0..b {
                    for r2 in 0..b {
                        sg[[r1, r2]] += z[r1] as f64 * z[r2] as f64;
                    }
                }
            }
        }

        self.rss += shard_rss;
        self.row_count += codes.len();
        Ok(BlockShardStats {
            rows: codes.len(),
            rss: shard_rss,
            alive_blocks: self.alive_count,
        })
    }

    /// end_epoch: refresh γ from the accumulated least-squares, refresh every
    /// touched block frame by a polar step over its accumulated cross-moment,
    /// revive dead blocks onto worst-reconstructed residual rows, capture the
    /// utilisation / stable-rank report, then reset the epoch accumulators.
    pub fn end_epoch(&mut self) -> Result<BlockEpochStats, String> {
        if self.row_count == 0 {
            return Err(
                "BlockSparseStream.end_epoch: no rows were streamed this epoch (call partial_fit \
                 with at least one shard first)"
                    .to_string(),
            );
        }
        let p = self.p;
        let b = self.b;

        // EV of the frames routed against this epoch, from the streamed moments.
        let n = self.row_count as f64;
        let mut tss = 0.0f64;
        for c in 0..p {
            tss += self.col_sumsq[c] - self.col_sum[c] * self.col_sum[c] / n;
        }
        let ev = if tss <= 1.0e-24 {
            if self.rss <= 1.0e-24 { 1.0 } else { 0.0 }
        } else {
            1.0 - self.rss / tss
        };

        // (γ) closed-form shared scalar from the accumulated least-squares.
        let gamma_new = if self.gamma_den <= 1.0e-24 {
            self.gamma
        } else {
            (self.gamma_num / self.gamma_den) as f32
        };

        // (frames) polar refresh of every touched block over its cross-moment.
        let ridge = self.config.frame_ridge;
        for gg in 0..self.g {
            if !self.touched[gg] {
                continue;
            }
            if ridge > 0.0 {
                for rr in 0..b {
                    for c in 0..p {
                        self.cross[gg][[c, rr]] += ridge * self.decoder[[gg * b + rr, c]] as f64;
                    }
                }
            }
            if let Ok(frame) = GrassmannFrame::polar_update(self.cross[gg].view()) {
                let u = frame.frame(); // P×b column-orthonormal
                let sv = frame.gauge_singular_values();
                let full_rank = sv.len() == b && sv.iter().all(|&s| s > 1.0e-9);
                if full_rank && u.ncols() == b {
                    for rr in 0..b {
                        for c in 0..p {
                            self.decoder[[gg * b + rr, c]] = u[[c, rr]] as f32;
                        }
                    }
                }
            }
        }

        // (revival) AuxK dead-block reseeding from worst-reconstructed rows.
        let dead: usize = self.usage.iter().filter(|&&u| u == 0).count();
        let revived = self.revive();

        // Utilisation + stable-rank report from this epoch's accumulators.
        for gg in 0..self.g {
            self.last_util[gg] = self.usage[gg] as f32 / self.row_count.max(1) as f32;
            self.last_stable[gg] = stable_rank_symmetric(self.second[gg].view());
        }

        self.gamma = gamma_new;
        let improve = ev - self.prev_ev;
        let converged =
            revived == 0 && improve.abs() <= self.config.tolerance && self.epochs_run > 0;

        self.prev_ev = ev;
        self.last_ev = ev;
        self.last_revived = revived;
        self.converged = converged;
        self.epochs_run += 1;
        let epoch = self.epochs_run;

        // Stash this (complete) epoch's accumulators for the certification
        // read surface (`block_rank_charges`) before the reset zeroes them.
        self.last_second.clone_from(&self.second);
        self.last_usage.clone_from(&self.usage);
        self.last_rss = self.rss;
        self.last_rows = self.row_count;

        self.reset_epoch();

        Ok(BlockEpochStats {
            explained_variance: ev,
            revived,
            dead,
            gamma: self.gamma,
            converged,
            epoch,
        })
    }

    /// Reseed each of the `k_aux` worst-utilised (dead) blocks with `b` orthonormal
    /// rows Gram–Schmidt'd from `b` distinct worst-reconstructed residual rows
    /// (never PCs, the house rule). Distinct contiguous groups of high-residual
    /// rows so revived blocks do not duplicate. Residuals are measured under the
    /// pre-refresh frames (the reservoir; see the module note).
    fn revive(&mut self) -> usize {
        if self.config.aux_k == 0 {
            return 0;
        }
        let ranked = self.reservoir.ranked();
        if ranked.is_empty() {
            return 0;
        }
        let b = self.b;
        let p = self.p;
        // Dead blocks (never fired) in ascending index order — the k_aux worst-
        // utilised, all at zero usage.
        let dead_blocks: Vec<usize> = (0..self.g).filter(|&gg| self.usage[gg] == 0).collect();

        let mut revived = 0usize;
        let mut cursor = 0usize;
        for &gg in dead_blocks.iter().take(self.config.aux_k) {
            if cursor + b > ranked.len() {
                break; // not enough distinct residual rows left to seed a frame
            }
            if ranked[cursor].norm2 <= DEAD_DENOM {
                break; // remaining rows already reconstructed — nothing to seed
            }
            let mut seed = Array2::<f32>::zeros((b, p));
            for rr in 0..b {
                let src = &ranked[cursor + rr].residual;
                for c in 0..p {
                    seed[[rr, c]] = src[c];
                }
            }
            cursor += b;
            gram_schmidt_rows(&mut seed);
            for rr in 0..b {
                for c in 0..p {
                    self.decoder[[gg * b + rr, c]] = seed[[rr, c]];
                }
            }
            revived += 1;
        }
        revived
    }

    fn reset_epoch(&mut self) {
        for mg in self.cross.iter_mut() {
            mg.fill(0.0);
        }
        for sg in self.second.iter_mut() {
            sg.fill(0.0);
        }
        for u in self.usage.iter_mut() {
            *u = 0;
        }
        for t in self.touched.iter_mut() {
            *t = false;
        }
        self.alive_count = 0;
        self.gamma_num = 0.0;
        self.gamma_den = 0.0;
        for c in 0..self.p {
            self.col_sum[c] = 0.0;
            self.col_sumsq[c] = 0.0;
        }
        self.rss = 0.0;
        self.row_count = 0;
        self.reservoir.clear();
    }

    /// finalize: hand back the warm-started block frames, γ, and run metadata,
    /// including the last epoch's per-block utilisation + stable-rank report. The
    /// routing is not materialised (a streamed corpus has no `N×k` object); route
    /// held-out or training shards back through the frozen frames to encode them.
    pub fn finalize(&self) -> BlockSparseStreamArtifact {
        BlockSparseStreamArtifact {
            decoder: self.decoder.clone(),
            gamma: self.gamma,
            block_topk: self.k,
            block_size: self.b,
            block_utilization: self.last_util.clone(),
            block_stable_rank: self.last_stable.clone(),
            epochs: self.epochs_run,
            explained_variance: self.last_ev,
            converged: self.converged,
        }
    }

    /// Per-block honest-charge ledger from the LAST CLOSED epoch (#23
    /// certification surface). For each block `g`: `d_eff` is the realised
    /// rank-charge DOF of its orthonormal frame `D_g` under the epoch's code
    /// Gram `C_g` (the SAME `realised_rank_charge_dof` currency the joint
    /// PROMOTE/DEMOTE gates charge); `delta_deviance = ½·tr(C_g)/φ̂` is the
    /// deviance reduction the block's codes claim (frames are block-
    /// orthonormal, so `tr(C_g)` is the energy the block reconstructs);
    /// `charge = ½·d_eff·ln(n_obs)`; `kept = margin > 0`. The dispersion
    /// `φ̂ = rss/(rows·p)` comes from the same closed epoch; a non-finite or
    /// non-positive `φ̂` falls back to the historical unit-dispersion reading
    /// (mirrors the hybrid-split #2124 guard). Errors if no epoch has closed.
    pub fn block_rank_charges(&self, n_obs: usize) -> Result<BlockRankCharges, String> {
        if self.last_rows == 0 {
            return Err(
                "block_rank_charges: no closed epoch to certify; call end_epoch first".to_string(),
            );
        }
        let phi_raw = self.last_rss / (self.last_rows as f64 * self.p as f64);
        let phi = if phi_raw.is_finite() && phi_raw > 0.0 {
            phi_raw
        } else {
            1.0
        };
        let ln_n = (n_obs.max(2) as f64).ln();
        let mut out = BlockRankCharges {
            block: Vec::with_capacity(self.g),
            n_eff: Vec::with_capacity(self.g),
            d_eff: Vec::with_capacity(self.g),
            delta_deviance: Vec::with_capacity(self.g),
            charge: Vec::with_capacity(self.g),
            margin: Vec::with_capacity(self.g),
            kept: Vec::with_capacity(self.g),
        };
        for gg in 0..self.g {
            let n_eff = self.last_usage[gg] as f64;
            let frame = self
                .decoder
                .slice(ndarray::s![gg * self.b..(gg + 1) * self.b, ..])
                .mapv(f64::from);
            let d_eff = crate::manifold::realised_rank_charge_dof(
                &self.last_second[gg],
                &frame,
                n_eff,
                self.p as f64,
                phi,
                0.0,
                None,
            )?;
            let mut tr = 0.0_f64;
            for i in 0..self.b {
                tr += self.last_second[gg][[i, i]];
            }
            let delta_deviance = 0.5 * tr / phi;
            let charge = 0.5 * d_eff * ln_n;
            let margin = delta_deviance - charge;
            out.block.push(gg);
            out.n_eff.push(n_eff);
            out.d_eff.push(d_eff);
            out.delta_deviance.push(delta_deviance);
            out.charge.push(charge);
            out.margin.push(margin);
            out.kept.push(margin > 0.0);
        }
        Ok(out)
    }

    /// Read-only view of the current warm-started frames (`K×P`, block-orthonormal).
    pub fn decoder(&self) -> ArrayView2<'_, f32> {
        self.decoder.view()
    }

    /// Current shared tied scalar γ.
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Block routing budget `k` in use (`min(block_topk, G)`).
    pub fn block_topk(&self) -> usize {
        self.k
    }

    /// Block size `b`.
    pub fn block_size(&self) -> usize {
        self.b
    }

    /// Epochs closed so far.
    pub fn epochs_run(&self) -> usize {
        self.epochs_run
    }
}

/// The artifact [`BlockSparseStreamState::finalize`] returns: the trained block
/// frames + γ + per-block report + run metadata. No `N×k` routing — the streamed
/// corpus is re-encoded shard-by-shard through the frozen frames, not held here.
#[derive(Clone, Debug)]
pub struct BlockSparseStreamArtifact {
    /// Block frames, `K×P` (`K = G·b`), each block's `b` rows orthonormal.
    pub decoder: Array2<f32>,
    /// Shared tied scalar γ.
    pub gamma: f32,
    /// Block routing budget `k` used.
    pub block_topk: usize,
    /// Block size `b`.
    pub block_size: usize,
    /// Per-block utilisation (last epoch), length `G`.
    pub block_utilization: Vec<f32>,
    /// Per-block within-block code stable rank (last epoch), length `G`.
    pub block_stable_rank: Vec<f32>,
    /// Epochs closed.
    pub epochs: usize,
    /// EV of the final epoch's pass (pre-refresh frames of the last epoch).
    pub explained_variance: f64,
    /// Whether the streaming loop met the convergence rule.
    pub converged: bool,
}

fn validate_config(config: &BlockSparseConfig) -> Result<(), String> {
    if config.n_blocks == 0 {
        return Err("BlockSparseStream requires n_blocks >= 1".to_string());
    }
    if config.block_size == 0 {
        return Err("BlockSparseStream requires block_size >= 1".to_string());
    }
    if config.block_topk == 0 {
        return Err("BlockSparseStream requires block_topk >= 1".to_string());
    }
    if config.max_epochs == 0 {
        return Err("BlockSparseStream requires max_epochs >= 1".to_string());
    }
    if !(config.frame_ridge.is_finite() && config.frame_ridge >= 0.0) {
        return Err("BlockSparseStream frame_ridge must be finite and non-negative".to_string());
    }
    if !config.tolerance.is_finite() {
        return Err("BlockSparseStream tolerance must be finite".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod test_support {
    use super::BlockSparseStreamState;

    pub(super) trait ZeroBlockForTest {
        fn zero_block_for_test(&mut self, block: usize);
    }

    impl ZeroBlockForTest for BlockSparseStreamState {
        fn zero_block_for_test(&mut self, block: usize) {
            for r in 0..self.b {
                for c in 0..self.p {
                    self.decoder[[block * self.b + r, c]] = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "block_stream_tests.rs"]
mod block_stream_tests;
