//! Partial-fit streaming surface for the block-sparse lane (#1026 block extension).
//!
//! The one-shot [`super::fit_block_sparse_dictionary`] holds the whole `N×P`
//! corpus in memory and alternates route → γ-refresh → frame-refresh → revive over
//! it. For a real corpus (a 30M-row residual-stream harvest) the rows never fit at
//! once, so this module exposes an accumulated-moment alternation as a resumable handle a Python
//! loop drives one shard at a time — mirroring [`super::SparseDictStreamState`]:
//!
//! ```text
//! state = BlockSparseStreamState::new(seed, config)     // fit_begin
//! for _epoch in 0..max_epochs {
//!     for shard in shards { state.partial_fit(shard) }   // route + accumulate
//!     state.end_epoch()                                  // γ + frames + birth transaction
//! }
//! state.finalize()                                       // frames + metadata
//! ```
//!
//! All heavy state lives here, native-side: the warm-started block frames, the
//! epoch's accumulated per-block MOD cross-moments (`M_g`, `P×b`), the streaming γ
//! numerator/denominator, per-block usage + within-block code second moments (for
//! the utilisation / stable-rank report), the streaming TSS/RSS moments, and the
//! worst-reconstructed-row reservoir feeding AuxK dead-block birth proposals. A shard
//! round-trips only its own rows through Python — never the `K×P` frames or any
//! `N×K` object — so per-shard overhead is `O(shard × P)`, independent of `K` and
//! of the corpus length.
//!
//! During an epoch the block frames and the shared
//! scalar γ are FROZEN at their epoch-start values; every shard is routed against
//! them and its cross-moment / γ / moment contributions are summed (all additive),
//! so shard boundaries do not change the accumulated problem. Gamma-free data
//! and overlap moments let the frame step use the newly fitted γ without
//! replaying the corpus. The simultaneous polar updates minimize a quadratic
//! upper bound derived from the selected-block count, which prevents co-fired
//! blocks from overcorrecting against one another at fixed codes. One-shot uses
//! sequential block updates; the streaming trajectory is different.
//!
//! EV describes the pass's measured frames with their profiled γ. Proposed frames
//! remain an uncertified checkpoint until another pass measures them. Finalize
//! requires EV, γ and projector closure and returns the exact measured frames,
//! γ and EV; an EV coincidence alone never certifies an unmeasured proposal.

use super::BlockSparseConfig;
use super::block::{
    block_birth_evidence_margin, frame_fixed_point_residual, gram_schmidt_rows,
    reconstruct_stored_code_row, relative_scalar_change, route_and_code_all, seed_frames,
    stable_rank_symmetric,
};
use super::residual_reservoir::ResidualReservoir;
use super::update::{DEAD_DENOM, DecoderSolveStats};
use crate::frames::GrassmannFrame;
use ndarray::{Array2, ArrayView2};

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
    /// with their profiled gamma, from the streamed TSS/RSS moments.
    pub explained_variance: f64,
    /// Residual-row block births accepted after a complete candidate-vs-baseline
    /// streaming pass proved strict RSS improvement.
    pub accepted_births: usize,
    /// Whether one candidate frame is staged for exact candidate-vs-baseline
    /// adjudication on the next streamed pass.
    pub birth_pending: bool,
    /// Dead blocks detected this epoch (fired for no row before proposal).
    pub dead: usize,
    /// Refreshed shared tied scalar γ after this epoch.
    pub gamma: f32,
    /// Relative displacement from the pass's gamma to its conditional optimum.
    pub gamma_residual: f64,
    /// Relative projector displacement to the frame update at that same gamma.
    pub frame_residual: f64,
    /// Whether EV, gamma and frame-projector residuals meet the tolerance,
    /// with no accepted or pending block birth.
    pub converged: bool,
    /// Epochs completed so far (this one inclusive).
    pub epoch: usize,
    /// Solve certificate placeholder. The block lane refreshes its small `b×b`
    /// frames by an exact polar step (no matrix-free CG/percolation solve), so this
    /// carries the default (zeroed) certificate; the CG/percolation stats are the
    /// atom/dict lane's ([`super::update::DecoderSolveStats`]).
    pub decoder_solve_stats: DecoderSolveStats,
}

/// A streaming birth is a two-pass transaction. The candidate frame is active
/// for the next streamed pass while this object retains the complete pre-birth
/// decoder/gamma and accumulates the exact baseline RSS on the same rows. At
/// `end_epoch` the candidate commits only if it was selected and strictly lowers
/// full-pass RSS; otherwise the retained state is restored byte-for-byte.
struct PendingBlockBirth {
    block: usize,
    baseline_decoder: Array2<f32>,
    baseline_gamma: f32,
    baseline_rss: f64,
    baseline_rows: usize,
    baseline_usage: Vec<usize>,
    baseline_second: Vec<Array2<f64>>,
}

/// Resumable state for a streaming block-sparse fit. Construct with [`Self::new`]
/// (fit_begin), feed shards with [`Self::partial_fit`], close each epoch with
/// [`Self::end_epoch`], and read the frames out with [`Self::finalize`]. The block
/// frames, the shared scalar γ, and any pending birth transaction warm-start across every call.
pub struct BlockSparseStreamState {
    config: BlockSparseConfig,
    g: usize,
    b: usize,
    k: usize,
    p: usize,
    decoder: Array2<f32>,
    gamma: f32,

    // ---- accumulators reset at each end_epoch (frozen frames/γ used to fill) ----
    second: Vec<Array2<f64>>, // gamma-free projection second moment (b×b)
    cross: Vec<Array2<f64>>,  // gamma-free other-block reconstruction × projection (P×b)
    data_cross: Vec<Array2<f64>>, // data × projection (P×b)
    usage: Vec<usize>,
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
    last_ev_residual: f64,
    last_gamma_residual: f64,
    last_frame_residual: f64,
    epochs_run: usize,
    last_accepted_births: usize,
    converged: bool,
    last_util: Vec<f32>,
    last_stable: Vec<f32>,
    last_decoder_solve_stats: DecoderSolveStats,
    // Last CLOSED epoch's accumulators, stashed by `end_epoch` before
    // `reset_epoch` zeroes the live ones — the certification read surface
    // (`block_rank_charges`) prices blocks from a COMPLETE epoch, never a
    // partially-filled one.
    last_second: Vec<Array2<f64>>,
    last_usage: Vec<usize>,
    last_rss: f64,
    last_rows: usize,
    pending_birth: Option<PendingBlockBirth>,
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
    /// `delta_deviance − charge`: the descriptive held-out BIC margin for the
    /// block (positive ⇒ its codes claim more deviance reduction than their
    /// information charge). This is a model-selection score, NOT a p-value,
    /// e-value, or FDR-controlled discovery — a BIC margin `M` is not a valid
    /// log-e-value (`E[exp M] > 1` under the null), so it must never be fed to an
    /// e-BH certificate as a `log_e_value`. `kept` applies the descriptive
    /// `margin > 0` gate, the same convention as `block_chart::ChartEvidence`.
    pub margin: Vec<f64>,
    /// `margin > 0`.
    pub kept: Vec<bool>,
}

impl BlockSparseStreamState {
    /// fit_begin: seed the block frames from `seed` (a representative sample) and
    /// prime the epoch accumulators. The seed fixes `P` and the initial
    /// orthonormal frames (`seed_frames`); the corpus is streamed later through
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
            second: (0..g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
            cross: (0..g).map(|_| Array2::<f64>::zeros((p, b))).collect(),
            data_cross: (0..g).map(|_| Array2::<f64>::zeros((p, b))).collect(),
            usage: vec![0; g],
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
            last_ev_residual: f64::INFINITY,
            last_gamma_residual: f64::INFINITY,
            last_frame_residual: f64::INFINITY,
            epochs_run: 0,
            last_accepted_births: 0,
            converged: false,
            last_util: vec![0.0; g],
            last_stable: vec![0.0; g],
            last_decoder_solve_stats: DecoderSolveStats::default(),
            last_second: (0..g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
            last_usage: vec![0; g],
            last_rss: 0.0,
            last_rows: 0,
            pending_birth: None,
        })
    }

    /// fit_begin with caller-supplied block frames. This is the large-`K` front
    /// door for experiments that cannot afford farthest-point seeding over
    /// `K*N*P`; the supplied decoder is still required to be a `KxP` block
    /// dictionary with `K = n_blocks*block_size`, and every row must be finite.
    pub fn new_with_decoder(
        decoder: Array2<f32>,
        config: &BlockSparseConfig,
    ) -> Result<Self, String> {
        validate_config(config)?;
        if decoder.nrows() != config.n_blocks * config.block_size {
            return Err(format!(
                "BlockSparseStream decoder rows must equal n_blocks*block_size = {}, got {}",
                config.n_blocks * config.block_size,
                decoder.nrows()
            ));
        }
        if decoder.ncols() == 0 {
            return Err("BlockSparseStream decoder must have at least one column".to_string());
        }
        if !decoder.iter().all(|v| v.is_finite()) {
            return Err("BlockSparseStream decoder must be finite".to_string());
        }
        if config.block_size > decoder.ncols() {
            return Err(format!(
                "BlockSparseStream block_size b={} cannot exceed P={}",
                config.block_size,
                decoder.ncols()
            ));
        }

        let p = decoder.ncols();
        let g = config.n_blocks;
        let b = config.block_size;
        let k = config.block_topk.min(g).max(1);
        let cap = config.aux_k.saturating_mul(b).max(1);
        Ok(Self {
            config: *config,
            g,
            b,
            k,
            p,
            decoder,
            gamma: 1.0,
            second: (0..g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
            cross: (0..g).map(|_| Array2::<f64>::zeros((p, b))).collect(),
            data_cross: (0..g).map(|_| Array2::<f64>::zeros((p, b))).collect(),
            usage: vec![0; g],
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
            last_ev_residual: f64::INFINITY,
            last_gamma_residual: f64::INFINITY,
            last_frame_residual: f64::INFINITY,
            epochs_run: 0,
            last_accepted_births: 0,
            converged: false,
            last_util: vec![0.0; g],
            last_stable: vec![0.0; g],
            last_decoder_solve_stats: DecoderSolveStats::default(),
            last_second: (0..g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
            last_usage: vec![0; g],
            last_rss: 0.0,
            last_rows: 0,
            pending_birth: None,
        })
    }

    /// partial_fit: route + tied-code one shard against the FROZEN epoch frames/γ
    /// and fold its contributions into this epoch's accumulators. Reuses the exact
    /// block-tiled router/coder of the one-shot lane (`route_and_code_all`), so
    /// streaming the shards yields the same accumulated sparse MOD / γ system as
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
        // A new nonempty pass has no convergence evidence until it closes.
        self.converged = false;

        // Per-shard wall-clock start (#2227). The block lane processes one shard
        // per `partial_fit`; the epoch-level telemetry only lands at `end_epoch`,
        // so a shard that routes or codes slowly (a device stall on a future
        // GPU-wired route, or a pathological CPU GEMM) is otherwise silent until
        // the whole epoch finishes. Emitting a bounded, per-shard heartbeat makes
        // any such stall visible within one shard instead of one epoch.
        let shard_start = std::time::Instant::now();
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
        )?;
        // A staged birth is evaluated against the COMPLETE pre-birth model on
        // exactly the same shards. This shadow route is the information a true
        // streaming transaction needs: the corpus is unavailable at end_epoch,
        // so deciding from the residual reservoir alone would be a surrogate,
        // not the exact training criterion.
        let baseline_codes = match self.pending_birth.as_ref() {
            Some(pending) => Some(route_and_code_all(
                shard,
                pending.baseline_decoder.view(),
                pending.baseline_gamma,
                self.g,
                b,
                self.k,
                self.config.minibatch,
                self.config.block_tile,
            )?),
            None => None,
        };
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
            let mut sel: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::with_capacity(self.k);
            let mut proj_sum = vec![0.0_f64; p];
            for j in 0..code.blocks.len() {
                if code.gates[j] == 0.0 {
                    continue;
                }
                let gg = code.blocks[j] as usize;
                let mut w = vec![0.0_f64; b];
                for (rr, wr) in w.iter_mut().enumerate() {
                    let atom = self.decoder.row(gg * b + rr);
                    let mut acc = 0.0_f64;
                    for (xc, ac) in xi.iter().zip(atom.iter()) {
                        acc += *xc as f64 * *ac as f64;
                    }
                    *wr = acc;
                }
                let mut proj = vec![0.0_f64; p];
                for (rr, &wr) in w.iter().enumerate() {
                    if wr == 0.0 {
                        continue;
                    }
                    let atom = self.decoder.row(gg * b + rr);
                    for c in 0..p {
                        proj[c] += wr * atom[c] as f64;
                    }
                }
                for c in 0..p {
                    proj_sum[c] += proj[c];
                }
                sel.push((gg, w, proj));
            }

            // Full residual under the frozen model + streaming RSS/reservoir.
            let mut residual = vec![0.0f32; p];
            let mut norm2 = 0.0f64;
            for c in 0..p {
                let value = xi[c] as f64 - gamma as f64 * proj_sum[c];
                residual[c] = value as f32;
                norm2 += value * value;
            }
            shard_rss += norm2;
            if aux_on {
                self.reservoir
                    .offer(norm2, base_index + r as u64, residual.clone());
            }

            // Streaming γ least-squares: γ* = (Σ⟨x,p⟩)/(Σ‖p‖²).
            for c in 0..p {
                self.gamma_num += xi[c] as f64 * proj_sum[c];
                self.gamma_den += proj_sum[c] * proj_sum[c];
            }

            // Gamma-free frame moments and within-block second moments. These
            // additive statistics support the coordinated scalar update and
            // parallel polar majorizer in end_epoch without retaining rows.
            for (gg, w, proj) in sel.iter() {
                let gg = *gg;
                if self.usage[gg] == 0 {
                    self.alive_count += 1;
                }
                self.usage[gg] += 1;
                // Keep both gamma-free terms. The exact frozen-code polar
                // moment at ANY gamma is gamma*XW - gamma²*OtherProjectionW.
                // A moment formed using the pass's old gamma cannot be reused
                // after fitting gamma without changing the coordinate problem.
                let mg = &mut self.cross[gg];
                let xg = &mut self.data_cross[gg];
                for c in 0..p {
                    let other = proj_sum[c] - proj[c];
                    for (rr, &wr) in w.iter().enumerate() {
                        mg[[c, rr]] += other * wr;
                        xg[[c, rr]] += xi[c] as f64 * wr;
                    }
                }
                let sg = &mut self.second[gg];
                for r1 in 0..b {
                    for r2 in 0..b {
                        sg[[r1, r2]] += w[r1] * w[r2];
                    }
                }
            }
        }

        if let (Some(pending), Some(baseline_codes)) =
            (self.pending_birth.as_mut(), baseline_codes.as_ref())
        {
            for (row, code) in baseline_codes.iter().enumerate() {
                let reconstruction =
                    reconstruct_stored_code_row(code, pending.baseline_decoder.view(), b);
                for column in 0..p {
                    let residual = shard[[row, column]] - reconstruction[column];
                    pending.baseline_rss += residual as f64 * residual as f64;
                }
                for (slot, &block) in code.blocks.iter().enumerate() {
                    if code.gates[slot] == 0.0 {
                        continue;
                    }
                    let block = block as usize;
                    pending.baseline_usage[block] += 1;
                    let z = &code.codes[slot * b..slot * b + b];
                    for left in 0..b {
                        for right in 0..b {
                            pending.baseline_second[block][[left, right]] +=
                                z[left] as f64 * z[right] as f64;
                        }
                    }
                }
            }
            pending.baseline_rows += baseline_codes.len();
        }

        self.rss += shard_rss;
        self.row_count += codes.len();
        // Per-shard heartbeat (#2227): rows in this shard, cumulative rows, the
        // shard reconstruction RSS, live-block count, and the shard wall time. A
        // stalled shard stops advancing this line; under `RUST_LOG=info` a route
        // that never returns is diagnosable immediately rather than after a whole
        // silent epoch.
        log::info!(
            "[SAE block shard] rows={} total_rows={} rss={:.6e} alive_blocks={}/{} \
             shard_s={:.2}",
            codes.len(),
            self.row_count,
            shard_rss,
            self.alive_count,
            self.g,
            shard_start.elapsed().as_secs_f64(),
        );
        Ok(BlockShardStats {
            rows: codes.len(),
            rss: shard_rss,
            alive_blocks: self.alive_count,
        })
    }

    /// end_epoch: resolve any staged birth against candidate/baseline full-pass
    /// RSS, refresh γ and frames for the admitted state, stage at most one next
    /// residual-row proposal, capture the utilisation/stable-rank report, then
    /// reset the epoch accumulators.
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
        // Resolve a staged residual-row birth against the exact full-pass
        // criterion BEFORE any ordinary frame refresh consumes the candidate's
        // accumulators. A rejected proposal restores the complete baseline
        // decoder/gamma and its reporting accumulators; the candidate pass is
        // discarded. An accepted proposal remains live and can take the usual
        // gamma/frame coordinate step below.
        let mut accepted_births = 0usize;
        let mut rejected_birth = false;
        if let Some(pending) = self.pending_birth.take() {
            if pending.baseline_rows != self.row_count {
                return Err(format!(
                    "BlockSparseStream birth transaction saw {} candidate rows but {} baseline rows",
                    self.row_count, pending.baseline_rows,
                ));
            }
            let selected = self.usage[pending.block] > 0;
            let improvement_rss = pending.baseline_rss - self.rss;
            let evidence_margin = block_birth_evidence_margin(
                pending.block,
                improvement_rss,
                self.rss,
                self.usage[pending.block],
                &self.second[pending.block].mapv(|value| value * (self.gamma as f64).powi(2)),
                self.decoder.view(),
                self.row_count,
                self.p,
                self.b,
            )?;
            if selected && evidence_margin.is_some_and(|margin| margin > 0.0) {
                accepted_births = 1;
            } else {
                self.decoder = pending.baseline_decoder;
                self.gamma = pending.baseline_gamma;
                self.rss = pending.baseline_rss;
                self.usage = pending.baseline_usage;
                self.second = pending.baseline_second;
                self.alive_count = self.usage.iter().filter(|&&count| count > 0).count();
                rejected_birth = true;
            }
        }

        let previous_gamma = self.gamma;
        let mut candidate_decoder = self.decoder.clone();
        let mut gamma_residual = f64::INFINITY;
        let mut frame_residual = f64::INFINITY;
        if !rejected_birth {
            // (γ) closed-form shared scalar from the accumulated least-squares.
            self.gamma = if self.gamma_den == 0.0 {
                0.0
            } else {
                (self.gamma_num / self.gamma_den) as f32
            };
            if !self.gamma.is_finite() || self.gamma < 0.0 {
                return Err("BlockSparseStream gamma optimum is not finite and nonnegative".into());
            }
            gamma_residual = relative_scalar_change(previous_gamma, self.gamma);
            // Evaluate the scalar quadratic around the accurately accumulated
            // pass residual, rather than subtracting nearly equal total energies.
            let old = previous_gamma as f64;
            let gamma = self.gamma as f64;
            let correction =
                (gamma - old) * ((gamma + old) * self.gamma_den - 2.0 * self.gamma_num);
            let rss = self.rss + correction;
            let resolution = f64::EPSILON
                * (self.row_count * p) as f64
                * (self.rss.abs() + correction.abs() + self.col_sumsq.iter().sum::<f64>());
            if !rss.is_finite() || rss < -resolution {
                return Err(format!("BlockSparseStream profiled RSS is invalid: {rss}"));
            }
            self.rss = rss.max(0.0);
            for second in &mut self.second {
                second.mapv_inplace(|value| value * gamma * gamma);
            }

            // Form the frame proposal at the NEW gamma. Its moments use the
            // same frozen directions and routing as the scalar fit, with no
            // corpus replay and no old-gamma cross term left in the polar step.
            let ridge = self.config.frame_ridge;
            for gg in 0..self.g {
                if self.usage[gg] == 0 {
                    continue;
                }
                for rr in 0..b {
                    for c in 0..p {
                        self.cross[gg][[c, rr]] = gamma * self.data_cross[gg][[c, rr]]
                            - gamma * gamma * self.cross[gg][[c, rr]]
                            + ridge * self.decoder[[gg * b + rr, c]] as f64
                            + (self.k - 1) as f64
                                * (0..b)
                                    .map(|axis| {
                                        self.decoder[[gg * b + axis, c]] as f64
                                            * self.second[gg][[axis, rr]]
                                    })
                                    .sum::<f64>();
                    }
                }
                // ||sum_g delta_g||² <= k*sum_g ||delta_g||² majorizes
                // the simultaneous co-fired reconstruction change. The extra
                // (k-1)*D*second term makes these independent polar updates a
                // descent step for the fixed-code loss, rather than Jacobi
                // best responses that can counter-rotate and increase it.
                if self.cross[gg].iter().all(|&value| value == 0.0) {
                    continue;
                }
                let frame = GrassmannFrame::polar_update(self.cross[gg].view())
                    .map_err(|error| format!("BlockSparseStream polar block {gg}: {error}"))?;
                let u = frame.frame();
                for rr in 0..b {
                    // GrassmannFrame canonicalizes each column's sign for a
                    // span-valued caller. Restore the signed Procrustes
                    // orientation needed by the frozen codes: Q^T M is PSD.
                    let alignment: f64 = (0..p).map(|c| u[[c, rr]] * self.cross[gg][[c, rr]]).sum();
                    let orientation = if alignment < 0.0 { -1.0 } else { 1.0 };
                    for c in 0..p {
                        candidate_decoder[[gg * b + rr, c]] = (orientation * u[[c, rr]]) as f32;
                    }
                }
            }
            frame_residual = frame_fixed_point_residual(
                self.decoder.view(),
                candidate_decoder.view(),
                self.g,
                b,
            );
        }
        let ev = if tss <= 1.0e-24 {
            if self.rss <= 1.0e-24 { 1.0 } else { 0.0 }
        } else {
            1.0 - self.rss / tss
        };
        // The block lane's exact polar frames carry no matrix-free CG/percolation
        // certificate (that solver serves the atom/dict lane); report a default.
        let decoder_solve_stats = DecoderSolveStats::default();

        let dead: usize = self.usage.iter().filter(|&&u| u == 0).count();

        // Utilisation + stable-rank report from this epoch's accumulators.
        for gg in 0..self.g {
            self.last_util[gg] = self.usage[gg] as f32 / self.row_count.max(1) as f32;
            self.last_stable[gg] = stable_rank_symmetric(self.second[gg].view());
        }

        let improve = ev - self.prev_ev;
        let stationary = !rejected_birth
            && accepted_births == 0
            && improve.abs() <= self.config.tolerance
            && gamma_residual <= self.config.tolerance
            && frame_residual <= self.config.tolerance
            && self.epochs_run > 0;
        // Certify the frames actually measured in this pass, together with
        // their profiled gamma. A frame proposal needs the next pass before
        // it has either an EV or a gamma certificate of its own.
        if !stationary && !rejected_birth {
            self.decoder = candidate_decoder;
        }
        let birth_pending = !rejected_birth && self.stage_birth_proposal();
        let converged = stationary && !birth_pending;

        self.prev_ev = ev;
        self.last_ev = ev;
        self.last_ev_residual = improve.abs();
        self.last_gamma_residual = gamma_residual;
        self.last_frame_residual = frame_residual;
        self.last_accepted_births = accepted_births;
        self.converged = converged;
        self.last_decoder_solve_stats = decoder_solve_stats;
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
            accepted_births,
            birth_pending,
            dead,
            gamma: self.gamma,
            gamma_residual,
            frame_residual,
            converged,
            epoch,
            decoder_solve_stats,
        })
    }

    /// Stage one residual-row birth for exact adjudication on the NEXT streamed
    /// pass. The live decoder receives the candidate frame, while
    /// [`PendingBlockBirth`] owns the complete baseline decoder/gamma and the
    /// shadow accumulators needed to restore it. No birth is reported or treated
    /// as a parameter update until [`Self::end_epoch`] observes both nonzero
    /// routing and strict full-pass RSS improvement.
    fn stage_birth_proposal(&mut self) -> bool {
        if self.config.aux_k == 0 || self.pending_birth.is_some() {
            return false;
        }
        let Some(block) = (0..self.g)
            .filter(|&candidate| self.usage[candidate] == 0)
            .take(self.config.aux_k)
            .next()
        else {
            return false;
        };
        let b = self.b;
        let p = self.p;
        let proposal = {
            let ranked = self.reservoir.ranked();
            if ranked.len() < b || ranked[0].norm2 <= DEAD_DENOM {
                return false;
            }
            let mut seed = Array2::<f32>::zeros((b, p));
            for row in 0..b {
                for column in 0..p {
                    seed[[row, column]] = ranked[row].residual[column];
                }
            }
            gram_schmidt_rows(&mut seed);
            seed
        };

        let baseline_decoder = self.decoder.clone();
        let baseline_gamma = self.gamma;
        self.decoder
            .slice_mut(ndarray::s![block * b..block * b + b, ..])
            .assign(&proposal);
        self.pending_birth = Some(PendingBlockBirth {
            block,
            baseline_decoder,
            baseline_gamma,
            baseline_rss: 0.0,
            baseline_rows: 0,
            baseline_usage: vec![0; self.g],
            baseline_second: (0..self.g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
        });
        true
    }

    fn reset_epoch(&mut self) {
        for sg in self.second.iter_mut() {
            sg.fill(0.0);
        }
        for mg in self.cross.iter_mut() {
            mg.fill(0.0);
        }
        for moment in &mut self.data_cross {
            moment.fill(0.0);
        }
        for u in self.usage.iter_mut() {
            *u = 0;
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

    /// finalize: hand back the converged block frames, γ, and run metadata,
    /// including the last epoch's per-block utilisation + stable-rank report. The
    /// routing is not materialised (a streamed corpus has no `N×k` object); route
    /// held-out or training shards back through the frozen frames to encode them.
    ///
    /// A fit object must only ever come from a converged optimization (SPEC 20):
    /// if the streaming loop has not met the convergence rule, this is a typed
    /// error and the state itself remains the resumable checkpoint — stream more
    /// epochs and finalize again.
    pub fn finalize(&self) -> Result<BlockSparseStreamArtifact, String> {
        if !self.converged || self.pending_birth.is_some() {
            return Err(format!(
                "BlockSparseStream.finalize: streaming fit has not converged after {} epoch(s) \
                 (last EV {:.6e}, EV residual {:.3e}, gamma residual {:.3e}, frame residual {:.3e} \
                 vs tolerance {:.3e}, {} accepted block \
                 birth(s) in the last epoch, birth pending={}); the stream state is a resumable \
                 checkpoint, not a model — run more epochs until end_epoch reports convergence",
                self.epochs_run,
                self.last_ev,
                self.last_ev_residual,
                self.last_gamma_residual,
                self.last_frame_residual,
                self.config.tolerance,
                self.last_accepted_births,
                self.pending_birth.is_some(),
            ));
        }
        Ok(BlockSparseStreamArtifact {
            decoder: self.decoder.clone(),
            gamma: self.gamma,
            block_topk: self.k,
            block_size: self.b,
            block_utilization: self.last_util.clone(),
            block_stable_rank: self.last_stable.clone(),
            epochs: self.epochs_run,
            explained_variance: self.last_ev,
            decoder_solve_stats: self.last_decoder_solve_stats,
        })
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
    /// EV of these exact frames and gamma on the final epoch's corpus.
    pub explained_variance: f64,
    /// Solve certificate placeholder (default/zeroed): the block lane refreshes its
    /// small `b×b` frames by an exact polar step, not the matrix-free CG/percolation
    /// solver that serves the atom/dict lane.
    pub decoder_solve_stats: DecoderSolveStats,
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
#[path = "block_stream_tests.rs"]
mod block_stream_tests;
