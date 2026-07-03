//! Partial-fit streaming surface for the collapsed linear lane (#1026).
//!
//! The one-shot [`super::fit_sparse_dictionary`] holds the whole `N×P` corpus in
//! memory and alternates route → refresh → revive over it. For a real corpus
//! (`K ≈ 32_000` over tens of millions of tokens) the rows never fit at once, so
//! this module exposes the SAME alternation as a resumable handle that a Python
//! loop drives one shard at a time:
//!
//! ```text
//! state = SparseDictStreamState::new(seed, config)     // fit_begin
//! for _epoch in 0..max_epochs {
//!     for shard in shards { state.partial_fit(shard) }  // route + accumulate
//!     state.end_epoch()                                 // refresh + revive
//! }
//! state.finalize()                                      // decoder + metadata
//! ```
//!
//! All heavy state lives here, native-side: the warm-started decoder, the epoch's
//! accumulated decoder normal equations ([`DecoderNormalEq`]), the per-atom alive
//! mask, the streaming TSS/RSS moments, and the worst-reconstructed-row reservoir
//! that feeds dead-atom revival. A shard round-trips only its own rows through
//! Python — never the `K×P` decoder or any `N×K` object — so per-shard overhead
//! is `O(shard × P)`, independent of `K` and of the corpus length.
//!
//! **Equivalence to one-shot.** Each epoch's pass routes every shard against the
//! current decoder `D_{k-1}` and sums their `CᵀC`/`CᵀX` contributions
//! ([`DecoderNormalEq::accumulate`] is additive), so the assembled normal
//! equations — and therefore the refreshed `D_k` — are exactly those of a
//! full-batch refresh over the concatenation, up to f32 GEMM rounding. The one
//! deliberate difference from the one-shot loop is the revival residual: one-shot
//! reseeds dead atoms onto residuals measured under the POST-refresh decoder
//! `D_k`, whereas the streaming pass measures them under the decoder in force
//! during the pass, `D_{k-1}` (there is no second corpus pass to recompute them
//! against `D_k` without re-streaming every shard). Both are the same
//! "worst-reconstructed residual row" source — never principal components — and
//! the two coincide once the dictionary is fully populated and revival goes
//! quiescent.

use super::{ScoreRouteStats, SparseDictConfig};
use super::scoring::TileScorer;
use super::update::{
    DEAD_DENOM, DecoderNormalEq, route_and_code_all, seed_decoder, solve_decoder, unit_norm_rows,
};
use ndarray::{Array2, ArrayView2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Per-shard summary returned by [`SparseDictStreamState::partial_fit`], so the
/// driving loop can watch progress without materialising any decoder-sized array.
#[derive(Clone, Copy, Debug)]
pub struct ShardStats {
    /// Rows consumed from this shard.
    pub rows: usize,
    /// This shard's reconstruction residual energy `Σ ‖x − D c‖²` under the
    /// decoder in force this epoch (the pre-refresh decoder).
    pub rss: f64,
    /// Distinct atoms that have fired at least once so far this epoch (cumulative
    /// across the shards seen since the last [`SparseDictStreamState::end_epoch`]).
    pub alive_atoms: usize,
    /// CPU/GPU scoring counters for this shard route.
    pub score_route_stats: ScoreRouteStats,
}

/// Per-epoch summary returned by [`SparseDictStreamState::end_epoch`].
#[derive(Clone, Copy, Debug)]
pub struct EpochStats {
    /// Explained variance `1 − RSS/TSS` of the decoder that was routed against
    /// during this epoch's pass (i.e. the pre-refresh decoder), computed exactly
    /// from the streamed TSS/RSS moments over the epoch's rows.
    pub explained_variance: f64,
    /// Dead atoms revived onto worst-reconstructed residual rows this epoch.
    pub revived: usize,
    /// Dead atoms detected this epoch (fired for no row before revival).
    pub dead: usize,
    /// Whether the EV-improvement tolerance was met AND no atom was revived (the
    /// same stopping rule as the one-shot loop: never converge with a live tail).
    pub converged: bool,
    /// Epochs completed so far (this one inclusive).
    pub epoch: usize,
}

/// One candidate row for dead-atom revival: its residual vector (under the
/// pre-refresh decoder) and the energy used to rank it. Ordered so the
/// [`BinaryHeap`]'s max is the MOST-evictable entry (smallest energy, ties broken
/// toward the larger global index) — that keeps the reservoir holding the
/// worst-reconstructed rows with one-shot's deterministic tie-break (descending
/// energy, ascending row index).
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

/// Bounded reservoir of the worst-reconstructed rows seen this epoch. Capacity is
/// `K`: revival installs at most one atom per distinct row and there are at most
/// `K` dead atoms, so the top-`K` residual rows are all that can ever be needed.
/// Peak memory is `K×P` f32 — the decoder's own footprint, never `N×K`.
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

    /// Offer a row's residual to the reservoir. Rows already reconstructed (energy
    /// at or below the dead floor) can seed nothing and are dropped.
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

    fn clear(&mut self) {
        self.heap.clear();
    }

    /// Rows ranked for revival: descending residual energy, ties by ascending
    /// global index — one-shot's exact `revive_dead_atoms` order.
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

/// Resumable state for a streaming sparse-dictionary fit. Construct with
/// [`Self::new`] (fit_begin), feed shards with [`Self::partial_fit`], close each
/// epoch with [`Self::end_epoch`], and read the decoder out with
/// [`Self::finalize`]. The decoder and dead-atom revival state warm-start across
/// every call — nothing is re-seeded or re-validated per shard.
pub struct SparseDictStreamState {
    config: SparseDictConfig,
    s: usize,
    p: usize,
    decoder: Array2<f32>,
    scorer: TileScorer,

    // ---- accumulators reset at each end_epoch ----
    eq: DecoderNormalEq,
    alive: Vec<bool>,
    alive_count: usize,
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
    score_route_stats: ScoreRouteStats,
}

impl SparseDictStreamState {
    /// fit_begin: seed the decoder from `seed` (a representative sample — one
    /// shard, or the whole corpus for small problems) and prime the epoch
    /// accumulators. The seed fixes `P` and the initial atom directions
    /// (deterministic farthest-point, [`seed_decoder`]); the corpus itself is
    /// streamed later through [`Self::partial_fit`].
    pub fn new(seed: ArrayView2<'_, f32>, config: &SparseDictConfig) -> Result<Self, String> {
        validate_config(config)?;
        if seed.nrows() == 0 || seed.ncols() == 0 {
            return Err(
                "SparseDictStream requires a non-empty seed sample (N×P) to fix P and the initial \
                 atom directions"
                    .to_string(),
            );
        }
        if !seed.iter().all(|v| v.is_finite()) {
            return Err("SparseDictStream seed sample must be finite".to_string());
        }
        let k = config.n_atoms;
        let p = seed.ncols();
        let s = config.active.min(k).max(1);

        let mut decoder = seed_decoder(seed, k);
        unit_norm_rows(&mut decoder);

        let scorer = TileScorer::new(s, config.score_tile);
        Ok(Self {
            config: *config,
            s,
            p,
            decoder,
            scorer,
            eq: DecoderNormalEq::zeros(k, p),
            alive: vec![false; k],
            alive_count: 0,
            col_sum: vec![0.0; p],
            col_sumsq: vec![0.0; p],
            rss: 0.0,
            row_count: 0,
            reservoir: ResidualReservoir::new(k),
            prev_ev: f64::NEG_INFINITY,
            last_ev: f64::NEG_INFINITY,
            epochs_run: 0,
            last_revived: 0,
            converged: false,
            score_route_stats: ScoreRouteStats::default(),
        })
    }

    /// partial_fit: route + sparse-code one shard against the current decoder and
    /// fold its contributions into this epoch's accumulators. Reuses the exact
    /// minibatch router/coder of the one-shot lane ([`route_and_code_all`], GPU
    /// -offloaded above break-even), so streaming the shards of a corpus yields
    /// the same normal equations as one full-batch pass over the concatenation.
    pub fn partial_fit(&mut self, shard: ArrayView2<'_, f32>) -> Result<ShardStats, String> {
        if shard.nrows() == 0 {
            return Ok(ShardStats {
                rows: 0,
                rss: 0.0,
                alive_atoms: self.alive_count,
                score_route_stats: ScoreRouteStats::default(),
            });
        }
        if shard.ncols() != self.p {
            return Err(format!(
                "SparseDictStream.partial_fit: shard has P={} columns but the fit was begun with \
                 P={}",
                shard.ncols(),
                self.p
            ));
        }
        if !shard.iter().all(|v| v.is_finite()) {
            return Err("SparseDictStream.partial_fit shard must be finite".to_string());
        }

        let mut shard_route_stats = ScoreRouteStats::default();
        let codes = route_and_code_all(
            shard,
            self.decoder.view(),
            &self.scorer,
            self.s,
            self.config.code_ridge,
            self.config.minibatch,
            self.config.score_mode,
            Some(&mut shard_route_stats),
        )?;
        self.score_route_stats.absorb(shard_route_stats);

        // Decoder normal equations: CᵀC / CᵀX for this shard, summed into the
        // epoch's running system (exactly the one-shot assembly, one shard's rows
        // at a time).
        self.eq.accumulate(shard, &codes);

        // Alive mask, streaming TSS moments, per-row residual → RSS + reservoir.
        let base_index = self.row_count as u64;
        let mut shard_rss = 0.0f64;
        for (r, code) in codes.iter().enumerate() {
            let xi = shard.row(r);
            for c in 0..self.p {
                let v = xi[c] as f64;
                self.col_sum[c] += v;
                self.col_sumsq[c] += v * v;
            }
            let mut residual = vec![0.0f32; self.p];
            for c in 0..self.p {
                residual[c] = xi[c];
            }
            for j in 0..code.indices.len() {
                let cj = code.codes[j];
                if cj == 0.0 {
                    continue;
                }
                self.alive_mark(code.indices[j] as usize);
                let drow = self.decoder.row(code.indices[j] as usize);
                for c in 0..self.p {
                    residual[c] -= cj * drow[c];
                }
            }
            let mut norm2 = 0.0f64;
            for c in 0..self.p {
                norm2 += residual[c] as f64 * residual[c] as f64;
            }
            shard_rss += norm2;
            self.reservoir.offer(norm2, base_index + r as u64, residual);
        }

        self.rss += shard_rss;
        self.row_count += codes.len();
        Ok(ShardStats {
            rows: codes.len(),
            rss: shard_rss,
            alive_atoms: self.alive_count,
            score_route_stats: shard_route_stats,
        })
    }

    #[inline]
    fn alive_mark(&mut self, atom: usize) {
        if !self.alive[atom] {
            self.alive[atom] = true;
            self.alive_count += 1;
        }
    }

    /// end_epoch: refresh the decoder from the accumulated normal equations
    /// (exact block-diagonal solve), unit-norm, revive dead atoms onto the
    /// worst-reconstructed residual rows, unit-norm again, then reset the epoch
    /// accumulators. Returns the pre-refresh EV and the revival/convergence
    /// bookkeeping the driving loop needs to decide when to stop.
    pub fn end_epoch(&mut self) -> Result<EpochStats, String> {
        if self.row_count == 0 {
            return Err(
                "SparseDictStream.end_epoch: no rows were streamed this epoch (call partial_fit \
                 with at least one shard first)"
                    .to_string(),
            );
        }

        // EV of the decoder that was routed against this epoch, from the streamed
        // moments: TSS = Σ_c (Σx_c² − (Σx_c)²/n), RSS already accumulated. Exactly
        // the one-shot `explained_variance` over the same rows.
        let n = self.row_count as f64;
        let mut tss = 0.0f64;
        for c in 0..self.p {
            tss += self.col_sumsq[c] - self.col_sum[c] * self.col_sum[c] / n;
        }
        let ev = if tss <= 1.0e-24 {
            if self.rss <= 1.0e-24 { 1.0 } else { 0.0 }
        } else {
            1.0 - self.rss / tss
        };

        // (c) exact decoder refresh from this epoch's normal equations, then (d)
        // unit-norm. Dead atoms keep their current direction inside solve_decoder.
        solve_decoder(
            &mut self.decoder,
            &self.eq,
            self.config.decoder_ridge as f64,
        );
        unit_norm_rows(&mut self.decoder);

        // (e) dead-atom revival onto worst-reconstructed residual rows (never PCs).
        let dead: usize = self.alive.iter().filter(|&&a| !a).count();
        let revived = self.revive(dead);
        if revived > 0 {
            unit_norm_rows(&mut self.decoder);
        }

        // Same stopping rule as the one-shot loop: never converge while atoms are
        // still being revived (a large dictionary populates its tail over several
        // epochs); once quiescent, an EV plateau converges.
        let improve = ev - self.prev_ev;
        let converged =
            revived == 0 && improve.abs() <= self.config.tolerance && self.epochs_run > 0;

        self.prev_ev = ev;
        self.last_ev = ev;
        self.last_revived = revived;
        self.converged = converged;
        self.epochs_run += 1;
        let epoch = self.epochs_run;

        self.reset_epoch();

        Ok(EpochStats {
            explained_variance: ev,
            revived,
            dead,
            converged,
            epoch,
        })
    }

    /// Point each dead atom at a distinct worst-reconstructed row's residual
    /// direction (raw; the caller re-unit-norms). Mirrors one-shot's
    /// `revive_dead_atoms`: descending residual energy, ascending index, one atom
    /// per distinct row, skipping already-reconstructed rows. The residual is
    /// measured under the pre-refresh decoder (see the module note).
    fn revive(&mut self, dead: usize) -> usize {
        if dead == 0 {
            return 0;
        }
        let ranked = self.reservoir.ranked();
        if ranked.is_empty() {
            return 0;
        }
        let dead_atoms: Vec<usize> = (0..self.decoder.nrows())
            .filter(|&a| !self.alive[a])
            .collect();

        let mut revived = 0usize;
        for (t, &atom) in dead_atoms.iter().enumerate() {
            if t >= ranked.len() {
                break; // one atom per distinct row this epoch
            }
            let src = ranked[t];
            if src.norm2 <= DEAD_DENOM {
                break; // remaining rows are already reconstructed
            }
            let mut dst = self.decoder.row_mut(atom);
            for c in 0..self.p {
                dst[c] = src.residual[c];
            }
            revived += 1;
        }
        revived
    }

    fn reset_epoch(&mut self) {
        let k = self.decoder.nrows();
        self.eq = DecoderNormalEq::zeros(k, self.p);
        for a in self.alive.iter_mut() {
            *a = false;
        }
        self.alive_count = 0;
        for c in 0..self.p {
            self.col_sum[c] = 0.0;
            self.col_sumsq[c] = 0.0;
        }
        self.rss = 0.0;
        self.row_count = 0;
        self.reservoir.clear();
    }

    /// finalize: hand back the warm-started decoder and run metadata. The routing
    /// itself is not materialised here (a streamed corpus has no `N×s` object to
    /// return); route held-out or training shards back through
    /// [`super::sparse_dictionary_transform`] against this decoder to encode them.
    pub fn finalize(&self) -> SparseDictArtifact {
        SparseDictArtifact {
            decoder: self.decoder.clone(),
            active: self.s,
            epochs: self.epochs_run,
            explained_variance: self.last_ev,
            converged: self.converged,
            score_route_stats: self.score_route_stats,
        }
    }

    /// Read-only view of the current warm-started decoder (`K×P`, unit-norm rows).
    pub fn decoder(&self) -> ArrayView2<'_, f32> {
        self.decoder.view()
    }

    /// Active budget `s` actually in use (`min(active, K)`).
    pub fn active(&self) -> usize {
        self.s
    }

    /// Epochs closed so far.
    pub fn epochs_run(&self) -> usize {
        self.epochs_run
    }
}

/// The artifact [`SparseDictStreamState::finalize`] returns: the trained decoder
/// plus run metadata. Deliberately has no `N×s` routing — the streamed corpus is
/// re-encoded shard-by-shard through the frozen decoder, not held in the fit.
#[derive(Clone, Debug)]
pub struct SparseDictArtifact {
    /// Decoder, `K×P`, unit-norm rows.
    pub decoder: Array2<f32>,
    /// Active budget `s` used.
    pub active: usize,
    /// Epochs closed.
    pub epochs: usize,
    /// EV of the final epoch's pass (pre-refresh decoder of the last epoch); for a
    /// converged fit this equals the returned decoder's EV to tolerance.
    pub explained_variance: f64,
    /// Whether the streaming loop met the convergence rule.
    pub converged: bool,
    /// Aggregate CPU/GPU scoring counters across streamed shards.
    pub score_route_stats: ScoreRouteStats,
}

fn validate_config(config: &SparseDictConfig) -> Result<(), String> {
    if config.n_atoms == 0 {
        return Err("SparseDictStream requires K >= 1".to_string());
    }
    if config.active == 0 {
        return Err("SparseDictStream requires active (top_s) >= 1".to_string());
    }
    if config.max_epochs == 0 {
        return Err("SparseDictStream requires max_epochs >= 1".to_string());
    }
    if !(config.code_ridge.is_finite() && config.code_ridge >= 0.0) {
        return Err("SparseDictStream code_ridge must be finite and non-negative".to_string());
    }
    if !(config.decoder_ridge.is_finite() && config.decoder_ridge >= 0.0) {
        return Err("SparseDictStream decoder_ridge must be finite and non-negative".to_string());
    }
    if !config.tolerance.is_finite() {
        return Err("SparseDictStream tolerance must be finite".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod stream_tests {
    use super::{SparseDictConfig, SparseDictStreamState, TileScorer, route_and_code_all};
    use crate::sparse_dict::fit_sparse_dictionary;
    use ndarray::{Array2, ArrayView2};

    /// Deterministic synthetic corpus: `n` rows, each a scaled planted atom plus a
    /// small bleed into the next, so the fixed-K lane has real reconstructible
    /// structure (mirrors the Python `_planted` fixtures).
    fn planted(n: usize, k: usize, p: usize) -> Array2<f32> {
        let mut x = Array2::<f32>::zeros((n, p));
        for row in 0..n {
            let primary = row % k;
            let secondary = (primary + 1) % k;
            let scale = 0.7 + 0.01 * ((row / k) as f32);
            x[[row, primary % p]] += scale;
            x[[row, secondary % p]] += 0.2 * scale;
        }
        x
    }

    /// Explained variance of a decoder + fresh routing over `x` (the same quantity
    /// the trainer reports), computed directly so the test does not depend on which
    /// EV a given entry point caches.
    fn routed_ev(
        x: ArrayView2<'_, f32>,
        decoder: &Array2<f32>,
        s: usize,
        config: &SparseDictConfig,
    ) -> f64 {
        let scorer = TileScorer::new(s, config.score_tile);
        let codes = route_and_code_all(
            x,
            decoder.view(),
            &scorer,
            s,
            config.code_ridge,
            config.minibatch,
        )
        .expect("fresh route");
        let n = x.nrows();
        let p = x.ncols();
        let mut means = vec![0.0f64; p];
        for i in 0..n {
            for c in 0..p {
                means[c] += x[[i, c]] as f64;
            }
        }
        for m in means.iter_mut() {
            *m /= n as f64;
        }
        let mut rss = 0.0f64;
        let mut tss = 0.0f64;
        for (i, code) in codes.iter().enumerate() {
            let mut recon = vec![0.0f64; p];
            for j in 0..code.indices.len() {
                let cj = code.codes[j] as f64;
                if cj == 0.0 {
                    continue;
                }
                let drow = decoder.row(code.indices[j] as usize);
                for c in 0..p {
                    recon[c] += cj * drow[c] as f64;
                }
            }
            for c in 0..p {
                let r = x[[i, c]] as f64 - recon[c];
                rss += r * r;
                let t = x[[i, c]] as f64 - means[c];
                tss += t * t;
            }
        }
        if tss <= 1.0e-24 {
            if rss <= 1.0e-24 { 1.0 } else { 0.0 }
        } else {
            1.0 - rss / tss
        }
    }

    /// Drive the streaming surface over `shards` (in row order) for up to
    /// `max_epochs`, seeding from the concatenation, and return the final decoder.
    fn stream_fit(
        seed: ArrayView2<'_, f32>,
        shards: &[ArrayView2<'_, f32>],
        config: &SparseDictConfig,
    ) -> (Array2<f32>, usize) {
        let mut state = SparseDictStreamState::new(seed, config).expect("fit_begin");
        for _ in 0..config.max_epochs {
            for shard in shards {
                state.partial_fit(*shard).expect("partial_fit");
            }
            let stats = state.end_epoch().expect("end_epoch");
            if stats.converged {
                break;
            }
        }
        let artifact = state.finalize();
        (artifact.decoder, artifact.active)
    }

    #[test]
    fn streaming_over_shards_matches_one_shot_on_concatenation() {
        // Streaming the row-ordered shards of a corpus must reach the same fixed
        // point as a one-shot fit on the concatenation: the per-shard normal-eq
        // accumulation is additive, so the refresh sequence is identical up to f32
        // GEMM rounding.
        let (n, k, p) = (240usize, 6usize, 8usize);
        let x = planted(n, k, p);
        let config = SparseDictConfig {
            n_atoms: k,
            active: 1,
            minibatch: 32,
            max_epochs: 40,
            score_tile: 16,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-9,
        };

        let one_shot = fit_sparse_dictionary(x.view(), &config).expect("one-shot fit");

        // Four contiguous shards whose concatenation (row order) is exactly `x`.
        let chunk = n / 4;
        let shards: Vec<ArrayView2<'_, f32>> = (0..4)
            .map(|i| {
                let start = i * chunk;
                let end = if i == 3 { n } else { start + chunk };
                x.slice(ndarray::s![start..end, ..])
            })
            .collect();
        let (stream_decoder, s) = stream_fit(x.view(), &shards, &config);

        assert_eq!(
            stream_decoder.shape(),
            one_shot.decoder.shape(),
            "decoder shapes must match"
        );

        let ev_stream = routed_ev(x.view(), &stream_decoder, s, &config);
        assert!(
            (ev_stream - one_shot.explained_variance).abs() < 1.0e-3,
            "streamed EV {ev_stream} must match one-shot EV {} within 1e-3",
            one_shot.explained_variance
        );
        assert!(
            ev_stream > 0.9,
            "planted corpus should fit well, got EV {ev_stream}"
        );
    }

    /// Deterministic full-rank pseudo-random corpus in `[-1, 1)` (index hash), so
    /// an undercomplete dictionary reconstructs it only partially and the fit has
    /// real headroom to improve across epochs.
    fn pseudo_random(n: usize, p: usize) -> Array2<f32> {
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            for c in 0..p {
                let h = (i.wrapping_mul(73_856_093) ^ c.wrapping_mul(19_349_663)) as u64;
                let h = h.wrapping_mul(2_654_435_761) % 2_000;
                x[[i, c]] = h as f32 / 1_000.0 - 1.0;
            }
        }
        x
    }

    #[test]
    fn warm_start_persists_across_epochs() {
        // The decoder must warm-start across partial_fit/end_epoch calls: a later
        // epoch's pre-refresh EV (which sees the decoder refreshed by earlier
        // epochs) strictly improves on the first epoch's. Uses full-rank data with
        // an undercomplete K so the seed is far from a perfect fit.
        let (n, k, p) = (300usize, 8usize, 12usize);
        let x = pseudo_random(n, p);
        let config = SparseDictConfig {
            n_atoms: k,
            active: 1,
            minibatch: 64,
            max_epochs: 6,
            score_tile: 16,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-12,
        };
        let mut state = SparseDictStreamState::new(x.view(), &config).expect("fit_begin");
        let mut evs = Vec::new();
        for _ in 0..config.max_epochs {
            state.partial_fit(x.view()).expect("partial_fit");
            evs.push(state.end_epoch().expect("end_epoch").explained_variance);
        }
        assert!(
            evs[1] > evs[0] + 1.0e-4,
            "second-epoch EV {} must improve on first-epoch EV {} (warm-start persisted)",
            evs[1],
            evs[0]
        );
    }

    #[test]
    fn revival_targets_worst_reconstructed_row_not_pcs() {
        // Seed the decoder from a sample spanning only e0/e1 (so one seeded atom is
        // a redundant duplicate that fires for no shard row — a dead atom), then
        // stream a shard whose lone e2 row no atom can reconstruct — the worst
        // residual row. Revival must point the orphaned atom at that row's residual
        // direction (e2), which is the shard's least-variance axis, never a PC.
        let p = 4usize;
        let mut seed = Array2::<f32>::zeros((20, p));
        for i in 0..10 {
            seed[[i, 0]] = 3.0;
            seed[[10 + i, 1]] = 3.0;
        }
        let mut shard = Array2::<f32>::zeros((21, p));
        for i in 0..10 {
            shard[[i, 0]] = 3.0;
            shard[[10 + i, 1]] = 3.0;
        }
        shard[[20, 2]] = 2.0; // lone e2 row — unreconstructable by an e0/e1 decoder

        let config = SparseDictConfig {
            n_atoms: 3,
            active: 1,
            minibatch: 64,
            max_epochs: 5,
            score_tile: 16,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 0.0,
        };
        let mut state = SparseDictStreamState::new(seed.view(), &config).expect("fit_begin");
        // Seed spans only e0/e1: nothing points at e2 yet.
        let pre_cos_e2 = (0..3)
            .map(|a| state.decoder()[[a, 2]].abs())
            .fold(0.0f32, f32::max);
        assert!(
            pre_cos_e2 < 1.0e-4,
            "seed decoder must not span e2, got |cos|={pre_cos_e2}"
        );

        state.partial_fit(shard.view()).expect("partial_fit");
        let stats = state.end_epoch().expect("end_epoch");
        assert!(
            stats.dead >= 1 && stats.revived >= 1,
            "expected a dead atom revived; dead={} revived={}",
            stats.dead,
            stats.revived
        );

        // A revived atom now points at e2 — the worst-reconstructed row's residual
        // direction — where none did before.
        let post_cos_e2 = (0..3)
            .map(|a| state.decoder()[[a, 2]].abs())
            .fold(0.0f32, f32::max);
        assert!(
            post_cos_e2 > 0.999,
            "a revived atom must equal e2, got |cos|={post_cos_e2}"
        );
    }
}
