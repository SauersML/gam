//! Fixed-K sparse, minibatched SAE trainer (#1026, "collapsed linear lane").
//!
//! This is an **additive, standalone** path that makes very large dictionaries
//! (`K` up to tens of thousands) tractable, where the exact-REML / Arrow-Schur
//! dense joint manifold solver in [`crate::manifold`] is the wrong
//! engine: that solver carries a dense `N×K` latent state, `N×K×P` sensitivity
//! tensors, `K²N` penalty couplings, and a joint Newton over all `K` outer
//! parameters. None of that survives `K ≈ 32_000`.
//!
//! The collapsed linear lane instead trains a dictionary by alternating
//! minimisation with **no dense `N×K` object anywhere**:
//!
//! 1. **route** — for each row, score it against the whole dictionary in
//!    `K`-tiles ([`scoring`]) and keep only the top-`s` atoms online, so the
//!    `N×K` score matrix is produced one tile at a time and discarded;
//! 2. **codes** — solve the small `s×s` active-set least-squares system per row
//!    ([`codes`]), giving a fixed-width sparse code `(indices, codes)`;
//! 3. **decoder** — accumulate the sparse normal equations (method-of-optimal
//!    -directions / sparse GEMM) and refresh each atom ([`update`]);
//! 4. **project** — re-unit-norm every atom so the code scale is identified.
//!
//! All heavy state is FP32. The only dense `K`-sized objects are the decoder
//! itself (`K×P`) and the per-atom `P×P`/scalar accumulators — never `N×K`.
//!
//! The exact manifold engine is **untouched**: it remains the certification /
//! small-`K` path. This module is reached only through its own public entry
//! [`fit_sparse_dictionary`] (and the `gamfit` Python facade that wraps it).

mod block;
mod block_chart;
mod block_stream;
mod codes;
mod scoring;
#[cfg(target_os = "linux")]
mod scoring_gpu;
mod stream;
mod update;

#[cfg(test)]
mod tests;

pub use block::{
    BlockSparseConfig, BlockSparseFit, block_gates, block_projections_row,
    block_sparse_dictionary_block_coords, block_sparse_dictionary_lift_block,
    block_sparse_dictionary_project_residual, block_sparse_dictionary_transform,
    fit_block_sparse_dictionary, reconstruct_block_sparse_rows, reconstruct_row, route_row_blocks,
    row_loss,
};
pub use block_chart::{
    BlockChartComposeConfig, BlockChartComposeResult, BlockChartRecord, BlockSeedManifest,
    BlockSeedManifestConfig, BlockSeedRecord, ChartEvidence, MdlFeaturizerRow,
    block_sparse_dictionary_firings, block_sparse_dictionary_seed_manifest,
    compose_block_coordinate_charts,
};
pub use block_stream::{
    BlockEpochStats, BlockShardStats, BlockSparseStreamArtifact, BlockSparseStreamState,
};
pub use codes::SparseCode;
pub use scoring::{ScoreRoutePath, ScoreRouteResult, ScoreRouteStats, TileScorer, top_s_online};
#[cfg(target_os = "linux")]
pub use scoring_gpu::{
    DEVICE_SCORE_BLOCK_MIN_ELEMS, ScoreBlockPath, score_block_cpu, score_block_required,
};
pub use stream::{EpochStats, ShardStats, SparseDictArtifact, SparseDictStreamState};

use ndarray::{Array2, ArrayView2};

/// Shared (NOT per-atom) hyper-parameters for the collapsed linear lane.
///
/// The whole point of the sparse trainer is that `K` is too large to carry a
/// per-atom smoothing parameter / Newton state; every knob here is a single
/// scalar shared across the entire dictionary.
#[derive(Clone, Copy, Debug)]
pub struct SparseDictConfig {
    /// Dictionary width `K` (number of atoms).
    pub n_atoms: usize,
    /// Active budget `s`: how many atoms may fire per row (`top_s`). This is the
    /// shared routing-sparsity hyper-parameter.
    pub active: usize,
    /// Minibatch size (rows per route→code→accumulate step). The decoder is
    /// refreshed once per full epoch from the accumulated sparse normal
    /// equations, so this only bounds peak working set, not the solution.
    pub minibatch: usize,
    /// Number of full passes over the data.
    pub max_epochs: usize,
    /// Column tile width used when scoring rows against the dictionary. Score
    /// tiles of shape `minibatch × tile` are formed and discarded; the `N×K`
    /// score matrix is never materialised.
    pub score_tile: usize,
    /// Shared ridge on the per-row active-set code solve (Tikhonov on the
    /// `s×s` Gram). Identifies the codes when active atoms are collinear.
    pub code_ridge: f32,
    /// Shared ridge on the per-atom decoder refresh (method-of-optimal
    /// -directions normal equations). Keeps a thinly-used atom well posed.
    pub decoder_ridge: f32,
    /// Relative explained-variance improvement below which training stops.
    pub tolerance: f64,
    /// Per-fit score routing residency contract. `Required` is fail-closed: a
    /// high-`K` route that cannot run on the CUDA score-block path returns an
    /// error instead of silently scoring on the CPU.
    pub score_mode: gam_gpu::GpuMode,
}

impl SparseDictConfig {
    /// Construct a config for a `K`-atom dictionary, leaving every other knob at
    /// its shared default.
    pub fn new(n_atoms: usize) -> Self {
        Self {
            n_atoms,
            ..Self::default()
        }
    }
}

impl Default for SparseDictConfig {
    fn default() -> Self {
        Self {
            n_atoms: 1,
            active: 1,
            minibatch: 512,
            max_epochs: 30,
            score_tile: 4096,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-6,
            score_mode: gam_gpu::GpuMode::Auto,
        }
    }
}

/// Result of a collapsed-linear-lane fit.
///
/// The routing is stored fixed-width and **sparse**: `indices[N, s]` /
/// `codes[N, s]`. There is deliberately no dense `N×K` assignment matrix —
/// reconstructing it would defeat the purpose of the lane.
#[derive(Clone, Debug)]
pub struct SparseDictFit {
    /// Decoder, `K×P`, unit-norm rows (one atom per row).
    pub decoder: Array2<f32>,
    /// Active atom indices per row, `N×s` (column `j` of row `i` is the `j`-th
    /// active atom for that row). Rows with fewer than `s` live atoms pad with
    /// repeated indices whose matching code is zero.
    pub indices: Array2<u32>,
    /// Sparse codes per row, `N×s`, aligned with [`Self::indices`].
    pub codes: Array2<f32>,
    /// Final held-in explained variance (`1 − RSS/TSS`).
    pub explained_variance: f64,
    /// Number of epochs actually run.
    pub epochs: usize,
    /// Whether the EV-improvement tolerance was reached.
    pub converged: bool,
    /// Active budget `s` actually used (`min(active, K)`).
    pub active: usize,
    /// Aggregate CPU/GPU scoring counters over every route pass in the fit.
    pub score_route_stats: ScoreRouteStats,
}

impl SparseDictFit {
    /// Dense reconstruction `N×P` of the training rows from the sparse routing.
    ///
    /// This *does* allocate an `N×P` array (the data size, not `N×K`); it exists
    /// for diagnostics / EV checks, not as part of the trainer's hot loop.
    pub fn reconstruct(&self) -> Array2<f32> {
        reconstruct_sparse_rows(self.decoder.view(), self.indices.view(), self.codes.view())
            .expect("SparseDictFit stores internally validated routing")
    }
}

pub fn reconstruct_sparse_rows(
    decoder: ArrayView2<'_, f32>,
    indices: ArrayView2<'_, u32>,
    codes: ArrayView2<'_, f32>,
) -> Result<Array2<f32>, String> {
    if indices.dim() != codes.dim() {
        return Err(format!(
            "reconstruct_sparse_rows: indices shape {:?} does not match codes shape {:?}",
            indices.dim(),
            codes.dim()
        ));
    }
    let n = indices.nrows();
    let p = decoder.ncols();
    let mut out = Array2::<f32>::zeros((n, p));
    for i in 0..n {
        for j in 0..indices.ncols() {
            let atom = indices[[i, j]] as usize;
            if atom >= decoder.nrows() {
                return Err(format!(
                    "reconstruct_sparse_rows: atom index {atom} out of range 0..{}",
                    decoder.nrows()
                ));
            }
            let code = codes[[i, j]];
            if code == 0.0 {
                continue;
            }
            let row = decoder.row(atom);
            for c in 0..p {
                out[[i, c]] += code * row[c];
            }
        }
    }
    Ok(out)
}

/// Out-of-sample sparse-dictionary encode plus route-dispatch diagnostics.
#[derive(Clone, Debug)]
pub struct SparseDictTransform {
    /// Active atom indices per row, `M×active`.
    pub indices: Array2<u32>,
    /// Sparse codes per row, `M×active`.
    pub codes: Array2<f32>,
    /// CPU/GPU scoring counters for this transform route.
    pub score_route_stats: ScoreRouteStats,
}

/// Out-of-sample encode: route held-out rows `x` (`M×P`, f32) against a frozen
/// sparse dictionary `decoder` (`K×P`) and solve the per-row active-set ridge
/// codes, returning fixed-width `(indices, codes)` each `M×active`. This is the
/// OOS `transform` step for a fitted sparse dictionary — the tiled routing and
/// the active-set least squares both live in the Rust core, and the route step
/// uses the same GPU-dispatched high-`K` scorer as fitting.
pub fn sparse_dictionary_transform(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    active: usize,
    score_tile: usize,
    code_ridge: f32,
) -> Result<(Array2<u32>, Array2<f32>), String> {
    let transform = sparse_dictionary_transform_with_mode(
        x,
        decoder,
        active,
        score_tile,
        code_ridge,
        gam_gpu::gpu_mode(),
    )?;
    Ok((transform.indices, transform.codes))
}

/// Out-of-sample encode with an explicit score routing mode and route counters.
///
/// This is the Rust-native high-`K` T1 transform surface: callers that require
/// GPU scoring pass [`gam_gpu::GpuMode::Required`] and inspect
/// [`SparseDictTransform::score_route_stats`] to verify device engagement.
pub fn sparse_dictionary_transform_with_mode(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    active: usize,
    score_tile: usize,
    code_ridge: f32,
    score_mode: gam_gpu::GpuMode,
) -> Result<SparseDictTransform, String> {
    let k = decoder.nrows();
    if k == 0 {
        return Err("sparse_dictionary_transform: dictionary has no atoms".to_string());
    }
    if x.ncols() != decoder.ncols() {
        return Err(format!(
            "sparse_dictionary_transform: X has P={} columns but the decoder has P={}",
            x.ncols(),
            decoder.ncols()
        ));
    }
    let s = active.min(k).max(1);
    let scorer = TileScorer::new(s, score_tile.max(1));
    let routed = scorer.route_minibatch_with_mode(x, decoder, score_mode)?;
    let mut score_route_stats = ScoreRouteStats::default();
    score_route_stats.record_result(&routed);
    let m = x.nrows();
    let mut indices = Array2::<u32>::zeros((m, s));
    let mut codes = Array2::<f32>::zeros((m, s));
    for (row_idx, active_pairs) in routed.selections.iter().enumerate() {
        let code = codes::solve_row_codes(x.row(row_idx), decoder, active_pairs, s, code_ridge);
        for j in 0..s {
            indices[[row_idx, j]] = code.indices[j];
            codes[[row_idx, j]] = code.codes[j];
        }
    }
    Ok(SparseDictTransform {
        indices,
        codes,
        score_route_stats,
    })
}

/// Fit a fixed-`K` sparse minibatched linear dictionary to `x` (`N×P`).
///
/// This is the public entry of the collapsed linear lane. It never forms a
/// dense `N×K` object: scoring is tiled, routing is fixed-width sparse, and the
/// decoder is refreshed from accumulated sparse normal equations.
pub fn fit_sparse_dictionary(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
) -> Result<SparseDictFit, String> {
    update::run(x, config)
}
