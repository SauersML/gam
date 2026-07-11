//! Block-sparse dictionary lane (#1026 block extension, "block-TopK subspace
//! featurizer").
//!
//! This is an **additive, standalone** variant of the collapsed linear lane
//! ([`super`]) in which the `K` atoms are grouped into `G` **blocks** of `b`
//! atoms each (`K = G·b`, `b` small — typically 2–4), and routing selects whole
//! blocks rather than individual atoms. It is gam's response to Goodfire's
//! "Block-Sparse Featurizers": the two ideas the paper conflates — a block's
//! *presence* (is this subspace active?) and its *amplitude* (how much, in which
//! direction?) — are here kept strictly separate, and every routing / penalty
//! decision sees a block ONLY through its group ℓ₂ gate `‖z_g‖₂`, which makes the
//! whole objective invariant to the `O(b)` gauge of each block's internal basis.
//!
//! The model per row `x` (`ℝᵖ`):
//!
//! 1. **frames** — each block `g` owns a **column-orthonormal frame** `D_g`
//!    (`b×P`, `D_g D_gᵀ = I_b`): a point on the Stiefel manifold `St(b, P)`
//!    (equivalently its span is a point on the Grassmannian `Gr(b, P)`). The
//!    frame is re-orthonormalised each epoch by a closed-form polar step
//!    (reusing [`crate::frames::GrassmannFrame::polar_update`]).
//! 2. **tied encode** — the within-block code is the tied projection
//!    `z_g = γ · x D_gᵀ` (`ℝᵇ`), with ONE learned scalar `γ` shared across the
//!    whole dictionary. Because `D_g` is orthonormal, `‖x D_gᵀ‖₂` is exactly the
//!    energy `x` places in block `g`'s subspace.
//! 3. **gate** — the routing gate is `gate_g = ‖z_g‖₂ = γ‖x D_gᵀ‖₂`. This is the
//!    *presence* signal; the *amplitude/direction* lives in the signed vector
//!    `z_g`. They are decoupled by construction.
//! 4. **block-TopK route** — select the `k` blocks of largest gate. No ReLU: the
//!    codes are signed.
//! 5. **additive decode** — `x̂ = Σ_{g∈S} z_g D_g = γ Σ_{g∈S} x P_g`, a sum over
//!    the selected blocks' rank-`b` subspace projectors `P_g = D_gᵀ D_g`.
//!
//! **Gauge invariance (load-bearing).** Replacing a block's frame by `R D_g` for
//! any `R ∈ O(b)` sends `z_g → z_g Rᵀ` (so `‖z_g‖₂`, the gate, is unchanged) and
//! leaves the decode `z_g D_g` and hence the loss bit-unchanged. Every selection
//! and every reported quantity is a function of the gauge-invariant `‖z_g‖₂`
//! only. [`block_tests`] rotates a block basis by a random orthogonal matrix and
//! asserts identical selection and identical loss.
//!
//! **Training** is alternating minimisation, mirroring [`super::update`]:
//! encode+route every row → refresh the shared scalar `γ` in closed form → update
//! each block frame by a method-of-optimal-directions cross-moment followed by a
//! polar reprojection back onto the Stiefel manifold → revive dead blocks
//! (AuxK-style, seeded from the worst-reconstructed residual ROWS — never PCs, the
//! house rule) → re-encode and score EV for the stopping rule. No dense `N×K`
//! object is ever formed: routing is block-tiled exactly as the atom lane tiles
//! columns.

use super::scoring::TopSSelector;
use crate::frames::GrassmannFrame;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;
use std::fmt;

/// Typed failure from [`fit_block_sparse_dictionary`].
#[derive(Clone, Debug)]
pub enum BlockSparseFitError {
    InvalidInput {
        reason: String,
    },
    NumericalFailure {
        reason: String,
    },
    NonConvergence {
        epochs: usize,
        explained_variance: f64,
        ev_residual: f64,
        gamma_residual: f64,
        frame_residual: f64,
        routing_residual: f64,
        reconstruction_residual: f64,
        tolerance: f64,
        accepted_births: usize,
        polar_failures: usize,
    },
}

impl BlockSparseFitError {
    fn invalid_input(reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            reason: reason.into(),
        }
    }
}

impl From<String> for BlockSparseFitError {
    fn from(reason: String) -> Self {
        Self::NumericalFailure { reason }
    }
}

impl From<BlockSparseFitError> for String {
    fn from(error: BlockSparseFitError) -> Self {
        error.to_string()
    }
}

impl fmt::Display for BlockSparseFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { reason } | Self::NumericalFailure { reason } => {
                f.write_str(reason)
            }
            Self::NonConvergence {
                epochs,
                explained_variance,
                ev_residual,
                gamma_residual,
                frame_residual,
                routing_residual,
                reconstruction_residual,
                tolerance,
                accepted_births,
                polar_failures,
            } => write!(
                f,
                "fit_block_sparse_dictionary did not converge after {epochs} epochs: EV \
                 {explained_variance:.6}, EV residual {ev_residual:.3e}, gamma residual \
                 {gamma_residual:.3e}, frame-projector residual {frame_residual:.3e}, \
                 routing residual {routing_residual:.3e}, reconstruction residual \
                 {reconstruction_residual:.3e} (tolerance {tolerance:.3e}), accepted \
                 births {accepted_births}, polar failures {polar_failures}"
            ),
        }
    }
}

impl std::error::Error for BlockSparseFitError {}

/// Shared (NOT per-block) hyper-parameters for the block-sparse lane. As in
/// [`super::SparseDictConfig`] every knob is a single scalar shared across the
/// whole dictionary — the point is that `G` is too large to carry per-block
/// state.
#[derive(Clone, Copy, Debug)]
pub struct BlockSparseConfig {
    /// Number of blocks `G`. The dictionary has `K = G·block_size` atoms.
    pub n_blocks: usize,
    /// Block size `b`: atoms per block (the subspace dimension). Typically 2–4.
    pub block_size: usize,
    /// Block routing budget `k`: how many blocks may fire per row (block-TopK).
    pub block_topk: usize,
    /// Number of full passes over the data.
    pub max_epochs: usize,
    /// Minibatch size (rows per route step): bounds the peak routing working set.
    pub minibatch: usize,
    /// Block-tile width used when scoring rows against the dictionary. Score
    /// tiles of shape `minibatch × (block_tile·b)` are formed and discarded; the
    /// `N×K` score matrix is never materialised.
    pub block_tile: usize,
    /// Ridge on the per-block frame cross-moment refresh (Tikhonov on the polar
    /// step's cross-moment); keeps a thinly-used block's polar well posed.
    pub frame_ridge: f64,
    /// AuxK dead-block revival budget `k_aux`: at most this many worst-utilised
    /// (effectively dead) blocks are reseeded per epoch onto worst-residual rows.
    pub aux_k: usize,
    /// Flag-gated MATRYOSHKA-PREFIX readout. When enabled, the returned fit
    /// carries log-spaced prefix losses over the final block ordering so an
    /// MDL/spectrometer K-ladder can read `L(K)` from one nested artifact.
    pub matryoshka_prefix: bool,
    /// Relative explained-variance improvement below which training stops.
    pub tolerance: f64,
}

impl BlockSparseConfig {
    /// A config for `n_blocks` blocks of `block_size`, other knobs at default.
    pub fn new(n_blocks: usize, block_size: usize) -> Self {
        Self {
            n_blocks,
            block_size,
            ..Self::default()
        }
    }

    /// Construct a block dictionary from a **scalar** capacity and scalar
    /// per-row active budget.
    ///
    /// This is the comparison-safe constructor: `n_atoms` counts decoder rows
    /// and `active_atoms` counts active scalar coordinates, exactly as they do
    /// for a scalar TopK SAE.  Both quantities must partition into complete
    /// blocks.  There is deliberately no rounding or clamping: changing either
    /// quantity would launder capacity or sparsity through the block layout.
    pub fn from_scalar_budget(
        n_atoms: usize,
        active_atoms: usize,
        block_size: usize,
    ) -> Result<Self, String> {
        if block_size == 0 {
            return Err("block sparse scalar budget requires block_size >= 1".to_string());
        }
        if n_atoms == 0 {
            return Err("block sparse scalar budget requires n_atoms >= 1".to_string());
        }
        if active_atoms == 0 || active_atoms > n_atoms {
            return Err(format!(
                "block sparse scalar budget requires active_atoms in [1, {n_atoms}]; got {active_atoms}"
            ));
        }
        if n_atoms % block_size != 0 {
            return Err(format!(
                "block sparse scalar capacity K={n_atoms} is not divisible by block_size={block_size}"
            ));
        }
        if active_atoms % block_size != 0 {
            return Err(format!(
                "block sparse scalar active budget s={active_atoms} is not divisible by block_size={block_size}"
            ));
        }
        Ok(Self {
            n_blocks: n_atoms / block_size,
            block_size,
            block_topk: active_atoms / block_size,
            ..Self::default()
        })
    }

    /// Dictionary width `K = G·b`.
    pub fn n_atoms(&self) -> usize {
        self.n_blocks * self.block_size
    }

    /// Maximum active scalar coordinates per row, `block_topk * block_size`.
    pub fn active_atoms(&self) -> usize {
        self.block_topk * self.block_size
    }
}

impl Default for BlockSparseConfig {
    fn default() -> Self {
        Self {
            n_blocks: 1,
            block_size: 2,
            block_topk: 1,
            max_epochs: 30,
            minibatch: 512,
            block_tile: 1024,
            frame_ridge: 1.0e-9,
            aux_k: 0,
            matryoshka_prefix: false,
            tolerance: 1.0e-6,
        }
    }
}

/// Result of a block-sparse fit.
///
/// Routing is stored fixed-width and **sparse** at the block level: `blocks[N,k]`
/// (which blocks fired), `gates[N,k]` (their group ℓ₂ presence), and `codes[N,k,b]`
/// (the signed within-block amplitude). Presence (`gates`) and amplitude (`codes`)
/// are deliberately separate arrays — the decoupling the lane is built around.
#[derive(Clone, Debug)]
pub struct BlockSparseFit {
    /// Decoder, `K×P` (`K = G·b`), block `g` occupying rows `[g·b, g·b+b)`; each
    /// identified block's `b` rows are orthonormal (`D_g D_gᵀ = I_b`). The
    /// explicit all-zero-data boundary has a zero decoder because its subspaces
    /// are unidentifiable and `gamma = 0` makes its out-of-sample map identically
    /// zero.
    pub decoder: Array2<f32>,
    /// Selected block indices per row, `N×k`.
    pub blocks: Array2<u32>,
    /// Per-selected-block **gate** `‖z_g‖₂` (presence), `N×k`, aligned with
    /// [`Self::blocks`]. Rows with fewer than `k` live blocks pad with a zero gate.
    pub gates: Array2<f32>,
    /// Per-selected-block signed **within-block code** `z_g` (amplitude/direction),
    /// `N×k×b`, aligned with [`Self::blocks`].
    pub codes: Array3<f32>,
    /// Shared tied-encoder scalar `γ`.
    pub gamma: f32,
    /// Per-block utilisation: fraction of rows that selected each block, length `G`.
    pub block_utilization: Vec<f32>,
    /// Per-block stable rank of the within-block code second moment
    /// `C_g = Σ_i z_{ig} z_{ig}ᵀ` (`trace(C_g)/λ_max(C_g)`), length `G`. Reports the
    /// effective dimensionality each block actually uses (needed by the MDL lane);
    /// a block used along a single direction has stable rank → 1, one used fully
    /// across its `b` axes → `b`.
    pub block_stable_rank: Vec<f32>,
    /// Optional MATRYOSHKA-PREFIX loss ladder `(K_atoms, mean squared loss)`.
    /// Prefix sizes are log-spaced atom counts aligned to block boundaries
    /// (`K = prefix_blocks * block_size`) and include the full dictionary width.
    /// Empty when [`BlockSparseConfig::matryoshka_prefix`] is false.
    pub matryoshka_prefix_losses: Vec<(usize, f64)>,
    /// Final held-in explained variance (`1 − RSS/TSS`).
    pub explained_variance: f64,
    /// Number of epochs actually run.
    pub epochs: usize,
    /// Checkable fixed-point certificate for the final full alternation.
    pub convergence: BlockSparseConvergence,
    /// Block budget `k` actually used (`min(block_topk, G)`).
    pub block_topk: usize,
    /// Block size `b` actually used.
    pub block_size: usize,
}

/// Fixed-point evidence attached to every converged [`BlockSparseFit`].
#[derive(Clone, Copy, Debug)]
pub struct BlockSparseConvergence {
    /// Explained-variance displacement under one replayed full alternation.
    pub ev_residual: f64,
    /// Relative shared-scale displacement under that alternation.
    pub gamma_residual: f64,
    /// Maximum gauge-invariant projector displacement over all blocks.
    pub frame_residual: f64,
    /// Gauge-invariant displacement of the exposed block routing, measured from
    /// the selected blocks' code norms.
    pub routing_residual: f64,
    /// Reconstruction displacement relative to the input data energy.
    pub reconstruction_residual: f64,
    /// Accepted residual-row births in the replayed alternation. A successful
    /// certificate always records zero.
    pub accepted_births: usize,
    /// Failed polar subsolves in the replayed alternation. A successful
    /// certificate always records zero.
    pub polar_failures: usize,
    pub tolerance: f64,
}

impl BlockSparseConvergence {
    /// A certificate whose every residual is exactly zero against a positive
    /// tolerance, with no accepted births or polar failures — a trivially
    /// converged full-alternation fixed point. Used to mint [`BlockSparseFit`]
    /// values from fixed, hand-authored block routings.
    pub fn trivially_converged() -> Self {
        Self {
            ev_residual: 0.0,
            gamma_residual: 0.0,
            frame_residual: 0.0,
            routing_residual: 0.0,
            reconstruction_residual: 0.0,
            accepted_births: 0,
            polar_failures: 0,
            tolerance: 1e-6,
        }
    }
}

impl BlockSparseFit {
    /// Dense reconstruction `N×P` from the sparse block routing:
    /// `x̂_i = Σ_{g∈S_i} z_{ig} D_g`. Allocates the data-size `N×P`, not `N×K`.
    pub fn reconstruct(&self) -> Array2<f32> {
        let n = self.blocks.nrows();
        let p = self.decoder.ncols();
        let b = self.block_size;
        let mut out = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            for j in 0..self.block_topk {
                let g = self.blocks[[i, j]] as usize;
                for r in 0..b {
                    let code = self.codes[[i, j, r]];
                    if code == 0.0 {
                        continue;
                    }
                    let row = self.decoder.row(g * b + r);
                    for c in 0..p {
                        out[[i, c]] += code * row[c];
                    }
                }
            }
        }
        out
    }

    /// Read the MATRYOSHKA-PREFIX reconstruction loss at atom prefix `K`.
    ///
    /// `K` must be one of the logged prefix atom counts in
    /// [`Self::matryoshka_prefix_losses`]. The readout is intentionally exact:
    /// callers asking for an unlogged rung should choose the ladder up front
    /// rather than silently interpolating or refitting.
    pub fn read_loss_at_prefix(&self, k_atoms: usize) -> Result<f64, String> {
        self.matryoshka_prefix_losses
            .iter()
            .find(|&&(k, _)| k == k_atoms)
            .map(|&(_, loss)| loss)
            .ok_or_else(|| {
                format!(
                    "BlockSparseFit has no MATRYOSHKA-PREFIX loss at K={k_atoms}; logged prefixes: {:?}",
                    self.matryoshka_prefix_losses
                        .iter()
                        .map(|&(k, _)| k)
                        .collect::<Vec<_>>()
                )
            })
    }
}

// ---------------------------------------------------------------------------
// Gauge-invariant primitives (the load-bearing surface the tests pin).
// ---------------------------------------------------------------------------

/// Raw (γ-free) tied projection of one row onto every block:
/// `w_g = x D_gᵀ ∈ ℝᵇ` for each block `g`, flattened row-major into `G·b`. The
/// gate is `‖z_g‖₂ = γ‖w_g‖₂`; since `γ ≥ 0` is a shared scalar, ranking blocks
/// by gate is identical to ranking by `‖w_g‖₂`, so routing is `γ`-invariant.
pub fn block_projections_row(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    n_blocks: usize,
    b: usize,
) -> Array2<f32> {
    let mut w = Array2::<f32>::zeros((n_blocks, b));
    for g in 0..n_blocks {
        for r in 0..b {
            let atom = decoder.row(g * b + r);
            let mut acc = 0.0f32;
            for (xr, ar) in row.iter().zip(atom.iter()) {
                acc += *xr * *ar;
            }
            w[[g, r]] = acc;
        }
    }
    w
}

/// Group ℓ₂ gate `‖w_g‖₂` for every block from the raw projections `w`
/// (`G×b`). This is the sole quantity routing and the utilisation report see a
/// block through, so the whole objective is invariant to the `O(b)` gauge of the
/// block basis (`w_g → w_g Rᵀ` leaves `‖w_g‖₂` unchanged).
pub fn block_gates(w: ArrayView2<'_, f32>) -> Vec<f32> {
    w.outer_iter()
        .map(|wg| wg.iter().map(|v| v * v).sum::<f32>().sqrt())
        .collect()
}

/// Block-TopK routing: the indices of the `k` largest-gate blocks, sorted by
/// descending gate (ties by ascending block index, for determinism). Reuses the
/// atom lane's online top-`k` selector keyed by the (non-negative) gate.
pub fn route_row_blocks(gates: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut sel = TopSSelector::new(k.max(1));
    for (g, &gate) in gates.iter().enumerate() {
        sel.offer(g as u32, gate);
    }
    sel.finish()
}

/// Reconstruction of one row from a chosen block set under the frames `decoder`
/// and the tied scalar `γ`: `x̂ = γ Σ_{g∈sel} (x D_gᵀ) D_g`. Returns the dense
/// `ℝᵖ` reconstruction. Only the gauge-invariant subspace projector `P_g` of each
/// selected block enters, so `x̂` (and therefore the loss below) is invariant to
/// each block's internal `O(b)` basis.
pub fn reconstruct_row(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    selected: &[u32],
    gamma: f32,
    b: usize,
) -> Array1<f32> {
    let p = row.len();
    let mut out = Array1::<f32>::zeros(p);
    for &g in selected {
        let g = g as usize;
        // w_g = x D_gᵀ, then add γ · w_g D_g.
        for r in 0..b {
            let atom = decoder.row(g * b + r);
            let mut wr = 0.0f32;
            for (xr, ar) in row.iter().zip(atom.iter()) {
                wr += *xr * *ar;
            }
            let coef = gamma * wr;
            if coef == 0.0 {
                continue;
            }
            for c in 0..p {
                out[c] += coef * atom[c];
            }
        }
    }
    out
}

/// Squared reconstruction loss `‖x − x̂‖₂²` of one row under a block set — the
/// gauge-invariant per-row objective the tests compare across basis rotations.
pub fn row_loss(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    selected: &[u32],
    gamma: f32,
    b: usize,
) -> f64 {
    let recon = reconstruct_row(row, decoder, selected, gamma, b);
    row.iter()
        .zip(recon.iter())
        .map(|(&x, &r)| {
            let d = x as f64 - r as f64;
            d * d
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Frame orthonormalisation (Stiefel reprojection).
// ---------------------------------------------------------------------------

/// Re-orthonormalise one block's `b` rows in place so `D_g D_gᵀ = I_b`
/// (Stiefel reprojection). The closed-form polar factor
/// [`GrassmannFrame::polar_update`] of the block's `P×b` transpose is the nearest
/// column-orthonormal matrix in Frobenius norm; we transpose it back to `b×P`
/// row-orthonormal. A rank-deficient block (a collapsed / duplicated seed) is
/// repaired by modified Gram–Schmidt with a canonical-axis fallback so the frame
/// is always a genuine `St(b, P)` point.
pub(super) fn orthonormalize_block(block: &mut Array2<f32>) {
    let (b, p) = block.dim();
    assert!(b <= p, "block size b must not exceed output dim p");
    // Build the P×b transpose as an f64 cross-moment and polar it.
    let mut cm = Array2::<f64>::zeros((p, b));
    for r in 0..b {
        for c in 0..p {
            cm[[c, r]] = block[[r, c]] as f64;
        }
    }
    if let Ok(frame) = GrassmannFrame::polar_update(cm.view()) {
        let u = frame.frame(); // P×b, column-orthonormal
        // Verify the polar produced a full-rank orthonormal set; the smallest
        // gauge singular value collapsing to ~0 means a rank-deficient seed, which
        // polar cannot orthonormalise — fall through to Gram–Schmidt.
        let sv = frame.gauge_singular_values();
        let full_rank = sv.len() == b && sv.iter().all(|&s| s > 1.0e-9);
        if full_rank && u.ncols() == b {
            for r in 0..b {
                for c in 0..p {
                    block[[r, c]] = u[[c, r]] as f32;
                }
            }
            return;
        }
    }
    gram_schmidt_rows(block);
}

/// Modified Gram–Schmidt orthonormalisation of the rows in place, substituting a
/// canonical axis `e_j` for any row that collapses (so a rank-deficient seed
/// still yields `b` orthonormal rows). f64 accumulation, f32 storage.
pub(super) fn gram_schmidt_rows(block: &mut Array2<f32>) {
    let (b, p) = block.dim();
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(b);
    for r in 0..b {
        let mut v: Vec<f64> = (0..p).map(|c| block[[r, c]] as f64).collect();
        for u in basis.iter() {
            let dot: f64 = v.iter().zip(u).map(|(a, b)| a * b).sum();
            for (vc, uc) in v.iter_mut().zip(u) {
                *vc -= dot * uc;
            }
        }
        let mut norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 1.0e-9 {
            // Collapsed: pick the first canonical axis orthogonal to the basis.
            let mut installed = false;
            for axis in 0..p {
                let mut e = vec![0.0f64; p];
                e[axis] = 1.0;
                for u in basis.iter() {
                    let dot = u[axis];
                    for (ec, uc) in e.iter_mut().zip(u) {
                        *ec -= dot * uc;
                    }
                }
                let en = e.iter().map(|x| x * x).sum::<f64>().sqrt();
                if en > 1.0e-9 {
                    for ec in e.iter_mut() {
                        *ec /= en;
                    }
                    v = e;
                    norm = 1.0;
                    installed = true;
                    break;
                }
            }
            if !installed {
                // p < b would be needed to exhaust the axes; guarded by the caller.
                for c in 0..p {
                    v[c] = if c == r % p { 1.0 } else { 0.0 };
                }
                norm = 1.0;
            }
        }
        for vc in v.iter_mut() {
            *vc /= norm;
        }
        for c in 0..p {
            block[[r, c]] = v[c] as f32;
        }
        basis.push(v);
    }
}

// ---------------------------------------------------------------------------
// Encoding / routing over a corpus (block-tiled, never N×K).
// ---------------------------------------------------------------------------

/// One row's block routing: the selected block indices and their signed codes.
/// `pub(super)` so the streaming lane ([`super::block_stream`]) can consume the
/// same per-row codes the one-shot trainer produces.
#[derive(Clone)]
pub(super) struct RowBlockCode {
    /// Selected block indices, length `k` (padded with block 0 + zero gate/code
    /// when the row had fewer than `k` blocks with positive gate).
    pub(super) blocks: Vec<u32>,
    /// Gate `‖z_g‖₂` per selected block, length `k`.
    pub(super) gates: Vec<f32>,
    /// Signed within-block code `z_g = γ w_g`, `k×b` flattened row-major.
    pub(super) codes: Vec<f32>,
}

/// Route one minibatch `block_rows` (`B×P`) against the frames, scoring blocks a
/// **block-tile** at a time so peak score memory is `B × (block_tile·b)`, never
/// `N×K`. Returns each row's top-`k` `(block, gate)` shortlist. Mirrors
/// [`super::scoring::TileScorer::route_minibatch`] but the tile GEMM produces
/// per-block group ℓ₂ gates rather than per-atom scores.
pub(super) fn route_block_minibatch(
    block_rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    n_blocks: usize,
    b: usize,
    k: usize,
    block_tile: usize,
) -> Vec<Vec<(u32, f32)>> {
    let nb = block_rows.nrows();
    let mut selectors: Vec<TopSSelector> = (0..nb).map(|_| TopSSelector::new(k)).collect();
    let tile = block_tile.max(1);
    let mut g0 = 0usize;
    while g0 < n_blocks {
        let g1 = (g0 + tile).min(n_blocks);
        // Atom rows [g0·b, g1·b): a (tile·b)×P slab. Score block = B×(tile·b).
        let atom_lo = g0 * b;
        let atom_hi = g1 * b;
        let slab = decoder.slice(ndarray::s![atom_lo..atom_hi, ..]);
        let scores = block_rows.dot(&slab.t()); // B × ((g1-g0)·b)
        for (row_idx, srow) in scores.axis_iter(Axis(0)).enumerate() {
            for (local_g, g) in (g0..g1).enumerate() {
                let base = local_g * b;
                let mut e = 0.0f32;
                for r in 0..b {
                    let v = srow[base + r];
                    e += v * v;
                }
                selectors[row_idx].offer(g as u32, e.sqrt());
            }
        }
        g0 = g1;
    }
    selectors.into_iter().map(TopSSelector::finish).collect()
}

fn orphan_gate_floor(row: ArrayView1<'_, f32>, b: usize) -> f32 {
    let row_norm = row
        .iter()
        .map(|v| {
            let vv = *v as f64;
            vv * vv
        })
        .sum::<f64>()
        .sqrt() as f32;
    let projection_roundoff = ((row.len().max(1) * b.max(1)) as f32).sqrt() * f32::EPSILON;
    row_norm * projection_roundoff
}

/// Encode + route the whole corpus in minibatches. For each row: block-TopK route
/// (`gate = ‖x D_gᵀ‖₂`), then the tied signed code `z_g = γ x D_gᵀ` for the
/// selected blocks. Returns one [`RowBlockCode`] per row in global order.
pub(super) fn route_and_code_all(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
    n_blocks: usize,
    b: usize,
    k: usize,
    minibatch: usize,
    block_tile: usize,
) -> Result<Vec<RowBlockCode>, String> {
    let n = x.nrows();
    let batch = minibatch.max(1);
    let mut out: Vec<RowBlockCode> = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        let end = (start + batch).min(n);
        let mb = x.slice(ndarray::s![start..end, ..]);
        let routed = route_block_minibatch_dispatch(mb, decoder, n_blocks, b, k, block_tile)?;
        let mut coded: Vec<RowBlockCode> = mb
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(routed.into_par_iter())
            .map(|(row, shortlist)| {
                let best_gate = shortlist.first().map(|entry| entry.1).unwrap_or(0.0);
                if best_gate < orphan_gate_floor(row, b) {
                    code_row(row, decoder, gamma, b, k, &[])
                } else {
                    code_row(row, decoder, gamma, b, k, &shortlist)
                }
            })
            .collect();
        out.append(&mut coded);
        start = end;
    }
    Ok(out)
}

/// Route one minibatch's blocks, dispatching to the CUDA block-gate router when a
/// GPU policy asks for it and a CUDA runtime is actually present, and
/// to the CPU router ([`route_block_minibatch`]) otherwise.
///
/// The dispatch honours the process-wide [`gam_gpu::GpuPolicy`]. `Off` always
/// takes the exact CPU router. `Auto` uses the device when admitted and falls
/// back to the CPU oracle when it is unavailable, below break-even, or faults.
/// `Required` is fail-closed for every refusal, including an unavailable runtime
/// and a below-break-even shape. The device route carries the #2227
/// bounded-progress checkpoints, so a device stall surfaces as a tile-attributed
/// error instead of a silent hang.
fn route_block_minibatch_dispatch(
    mb: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    n_blocks: usize,
    b: usize,
    k: usize,
    block_tile: usize,
) -> Result<Vec<Vec<(u32, f32)>>, String> {
    #[cfg(target_os = "linux")]
    {
        let policy = gam_gpu::global_policy();
        if policy != gam_gpu::GpuPolicy::Off {
            let (selections, _path, _dtoh) =
                super::block_scoring_gpu::route_blocks_required(mb, decoder, b, k, policy)
                    .map_err(|err| err.to_string())?;
            return Ok(selections);
        }
    }
    #[cfg(not(target_os = "linux"))]
    if gam_gpu::global_policy() == gam_gpu::GpuPolicy::Required {
        return Err(
            "block-gate route GpuPolicy::Required: CUDA routing is only compiled on Linux"
                .to_string(),
        );
    }
    Ok(route_block_minibatch(
        mb, decoder, n_blocks, b, k, block_tile,
    ))
}

/// Fixed-width sparse code for one row from its `(block, gate)` shortlist: the
/// tied signed within-block code `z_g = γ x D_gᵀ` per selected block, padded to
/// width `k` (block 0, zero gate/code) when fewer than `k` blocks fired.
fn code_row(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
    b: usize,
    k: usize,
    shortlist: &[(u32, f32)],
) -> RowBlockCode {
    let mut blocks = Vec::with_capacity(k);
    let mut gates = Vec::with_capacity(k);
    let mut codes = Vec::with_capacity(k * b);
    for &(g, gate) in shortlist.iter().take(k) {
        blocks.push(g);
        gates.push(gate);
        let gg = g as usize;
        for r in 0..b {
            let atom = decoder.row(gg * b + r);
            let mut wr = 0.0f32;
            for (xr, ar) in row.iter().zip(atom.iter()) {
                wr += *xr * *ar;
            }
            codes.push(gamma * wr);
        }
    }
    while blocks.len() < k {
        blocks.push(0);
        gates.push(0.0);
        for _ in 0..b {
            codes.push(0.0);
        }
    }
    RowBlockCode {
        blocks,
        gates,
        codes,
    }
}

// ---------------------------------------------------------------------------
// γ refresh, frame refresh, revival, EV.
// ---------------------------------------------------------------------------

/// Per-row un-scaled projection sum `p_i = Σ_{g∈S_i} x_i P_g` (the reconstruction
/// with `γ = 1`), used both by the closed-form `γ` solve and residual/EV. Because
/// `D_g` is orthonormal, `x_i P_g = (x_i D_gᵀ) D_g`, formed from the stored raw
/// projection `z_{ig}/γ` — but we recompute directly from the frames to stay
/// `γ`-independent.
fn projection_sum_row(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: &[u32],
    gates: &[f32],
    b: usize,
) -> Array1<f32> {
    let p = row.len();
    let mut out = Array1::<f32>::zeros(p);
    for (j, &g) in blocks.iter().enumerate() {
        if gates[j] == 0.0 {
            continue; // padded slot
        }
        let gg = g as usize;
        for r in 0..b {
            let atom = decoder.row(gg * b + r);
            let mut wr = 0.0f32;
            for (xr, ar) in row.iter().zip(atom.iter()) {
                wr += *xr * *ar;
            }
            if wr == 0.0 {
                continue;
            }
            for c in 0..p {
                out[c] += wr * atom[c];
            }
        }
    }
    out
}

/// Decode one stored sparse row exactly as [`BlockSparseFit::reconstruct`] will.
/// Keeping the alternation and its certificate on the stored code values avoids
/// certifying a freshly re-derived tied projection while returning different
/// arrays.
fn reconstruct_stored_code_row(
    code: &RowBlockCode,
    decoder: ArrayView2<'_, f32>,
    b: usize,
) -> Array1<f32> {
    let mut out = Array1::<f32>::zeros(decoder.ncols());
    for (slot, &block) in code.blocks.iter().enumerate() {
        if code.gates[slot] == 0.0 {
            continue;
        }
        let block = block as usize;
        for r in 0..b {
            let value = code.codes[slot * b + r];
            if value == 0.0 {
                continue;
            }
            let atom = decoder.row(block * b + r);
            for column in 0..decoder.ncols() {
                out[column] += value * atom[column];
            }
        }
    }
    out
}

/// Closed-form refresh of the shared tied scalar
/// `γ* = (Σ_i ⟨x_i, p_i⟩) / (Σ_i ‖p_i‖²)`, where `p_i = Σ_{g∈S_i} x_i P_g`. This is
/// the exact least-squares `γ` given the current frames and routing (decode is
/// `γ p_i`). The exact no-projection boundary is the null scale `γ = 0`.
fn refresh_gamma(
    x: ArrayView2<'_, f32>,
    codes: &[RowBlockCode],
    decoder: ArrayView2<'_, f32>,
    b: usize,
) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (i, code) in codes.iter().enumerate() {
        let xi = x.row(i);
        let p_i = projection_sum_row(xi, decoder, &code.blocks, &code.gates, b);
        for c in 0..xi.len() {
            num += xi[c] as f64 * p_i[c] as f64;
            den += p_i[c] as f64 * p_i[c] as f64;
        }
    }
    if den == 0.0 { 0.0 } else { (num / den) as f32 }
}

/// Refresh every block frame by a method-of-optimal-directions cross-moment
/// followed by a polar reprojection back onto the Stiefel manifold.
///
/// With the codes `z_{ig}` held fixed and the frame constrained orthonormal, the
/// reconstruction loss `Σ_i ‖r_{ig} − z_{ig} D_g‖²` — where
/// `r_{ig} = x_i − (x̂_i − z_{ig} D_g)` is the residual attributed to block `g`
/// alone — has, up to the `‖z_{ig} D_g‖² = ‖z_{ig}‖²` term that is constant in the
/// orthonormal gauge, the orthogonal-Procrustes solution `D_g = polar(M_g)ᵀ` with
/// cross-moment `M_g = Σ_i r_{ig}ᵀ z_{ig}` (`P×b`). We accumulate `M_g` for every
/// block, add a tiny ridge on its Gram for a thinly-used block, and polar it via
/// [`GrassmannFrame::polar_update`]. Blocks that no row selected accumulate a zero
/// `M_g` and keep their current frame in place (they are revived separately).
fn refresh_frames(
    x: ArrayView2<'_, f32>,
    codes: &[RowBlockCode],
    decoder: &mut Array2<f32>,
    n_blocks: usize,
    b: usize,
    ridge: f64,
) -> usize {
    let p = x.ncols();
    // Cross-moments M_g (P×b), one per block.
    let mut cm: Vec<Array2<f64>> = (0..n_blocks)
        .map(|_| Array2::<f64>::zeros((p, b)))
        .collect();
    let mut touched = vec![false; n_blocks];

    for (i, code) in codes.iter().enumerate() {
        let xi = x.row(i);
        // Full reconstruction x̂_i under the current frames/γ.
        if code.gates.iter().all(|&gate| gate == 0.0) {
            continue;
        }
        let recon = reconstruct_stored_code_row(code, decoder.view(), b);
        for (j, &g) in code.blocks.iter().enumerate() {
            if code.gates[j] == 0.0 {
                continue;
            }
            let gg = g as usize;
            // z_{ig}: the stored signed within-block code.
            let z = &code.codes[j * b..j * b + b];
            // Block g's own contribution decode_g = Σ_r z[r] D_g[r].
            // r_{ig} = x_i − (x̂_i − decode_g) = x_i − x̂_i + decode_g.
            // Accumulate M_g += r_{ig} zᵀ (outer, P×b).
            let mg = &mut cm[gg];
            for c in 0..p {
                let mut decode_g_c = 0.0f32;
                for r in 0..b {
                    decode_g_c += z[r] * decoder[[gg * b + r, c]];
                }
                let resid_c = (xi[c] - recon[c] + decode_g_c) as f64;
                for r in 0..b {
                    mg[[c, r]] += resid_c * z[r] as f64;
                }
            }
            touched[gg] = true;
        }
    }

    let mut polar_failures = 0usize;
    for g in 0..n_blocks {
        if !touched[g] {
            continue;
        }
        // Ridge the cross-moment's Gram lightly by shrinking toward the current
        // frame: add ridge·D_gᵀ (P×b of the current orthonormal rows). This keeps
        // a block that saw only a handful of rows well posed without disturbing a
        // well-populated one.
        if ridge > 0.0 {
            for r in 0..b {
                for c in 0..p {
                    cm[g][[c, r]] += ridge * decoder[[g * b + r, c]] as f64;
                }
            }
        }
        match GrassmannFrame::polar_update(cm[g].view()) {
            Ok(frame) => {
                let u = frame.frame(); // P×b column-orthonormal
                let sv = frame.gauge_singular_values();
                let largest_sv = sv.first().copied().unwrap_or(0.0);
                let numerical_rank_floor = largest_sv * f64::EPSILON * p.max(b) as f64;
                let full_rank = sv.len() == b
                    && largest_sv.is_finite()
                    && sv
                        .iter()
                        .all(|&s| s.is_finite() && s > numerical_rank_floor);
                if full_rank && u.ncols() == b {
                    for r in 0..b {
                        for c in 0..p {
                            decoder[[g * b + r, c]] = u[[c, r]] as f32;
                        }
                    }
                } else {
                    polar_failures += 1;
                }
            }
            Err(_) => polar_failures += 1,
        }
        // A degenerate cross-moment (rank-deficient) leaves the block's current
        // (already orthonormal) frame in place; revival handles a truly dead block.
    }
    polar_failures
}

/// AuxK-style dead-block revival (house rule: seed from residual ROWS, never PCs).
///
/// Identify the `k_aux` **worst-utilised** blocks (fewest rows selected this
/// epoch); any that are effectively dead (utilisation below one row) are reseeded.
/// A revived block's `b` orthonormal rows are Gram–Schmidt orthonormalised from
/// the `b` worst-reconstructed residual ROWS (each dead block takes a distinct
/// contiguous group of high-residual rows, so revived blocks do not duplicate).
/// This is the block analogue of the atom lane's dead-feature resampling and of
/// the AuxK auxiliary-reconstruction loss, but installs a genuine `St(b, P)` frame
/// spanning the directions the model most fails to explain.
///
/// Returns the block indices whose residual-row birth proposals were installed.
struct RevivedBlockProposal {
    block: usize,
    previous_frame: Array2<f32>,
}

fn revive_dead_blocks(
    x: ArrayView2<'_, f32>,
    codes: &[RowBlockCode],
    decoder: &mut Array2<f32>,
    n_blocks: usize,
    b: usize,
    aux_k: usize,
) -> Vec<RevivedBlockProposal> {
    if aux_k == 0 {
        return Vec::new();
    }
    let n = x.nrows();
    let p = x.ncols();

    // Per-block usage count.
    let mut usage = vec![0usize; n_blocks];
    for code in codes.iter() {
        for (j, &g) in code.blocks.iter().enumerate() {
            if code.gates[j] != 0.0 {
                usage[g as usize] += 1;
            }
        }
    }

    // The k_aux worst-utilised blocks (ascending usage, ties by index).
    let mut order: Vec<usize> = (0..n_blocks).collect();
    order.sort_by(|&a, &c| usage[a].cmp(&usage[c]).then(a.cmp(&c)));
    let candidates: Vec<usize> = order
        .into_iter()
        .take(aux_k)
        .filter(|&g| usage[g] == 0) // only truly dead blocks are reseeded
        .collect();
    if candidates.is_empty() {
        return Vec::new();
    }

    // Per-row residual energy under the current model.
    let mut resid = Array2::<f32>::zeros((n, p));
    let mut resid_norm2 = vec![0.0f64; n];
    for i in 0..n {
        let xi = x.row(i);
        let code = &codes[i];
        let recon = reconstruct_stored_code_row(code, decoder.view(), b);
        let mut acc = 0.0f64;
        for c in 0..p {
            let rc = xi[c] - recon[c];
            resid[[i, c]] = rc;
            acc += rc as f64 * rc as f64;
        }
        resid_norm2[i] = acc;
    }

    // Rows by descending residual energy (ties ascending index).
    let mut row_order: Vec<usize> = (0..n).collect();
    row_order.sort_by(|&a, &c| {
        resid_norm2[c]
            .partial_cmp(&resid_norm2[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&c))
    });

    let mut revived = Vec::new();
    let mut cursor = 0usize;
    for &g in candidates.iter() {
        // Take the next b distinct high-residual rows for this block's frame.
        if cursor >= n || resid_norm2[row_order[cursor]] == 0.0 {
            break; // no residual left to seed from
        }
        let mut seed = Array2::<f32>::zeros((b, p));
        for r in 0..b {
            let row = if cursor < n {
                row_order[cursor]
            } else {
                row_order[n - 1]
            };
            cursor += 1;
            for c in 0..p {
                seed[[r, c]] = resid[[row, c]];
            }
        }
        gram_schmidt_rows(&mut seed);
        let previous_frame = decoder.slice(ndarray::s![g * b..g * b + b, ..]).to_owned();
        for r in 0..b {
            for c in 0..p {
                decoder[[g * b + r, c]] = seed[[r, c]];
            }
        }
        revived.push(RevivedBlockProposal {
            block: g,
            previous_frame,
        });
    }
    revived
}

/// Held-in explained variance `1 − RSS/TSS` of the block reconstruction.
fn explained_variance(
    x: ArrayView2<'_, f32>,
    codes: &[RowBlockCode],
    decoder: ArrayView2<'_, f32>,
    b: usize,
) -> f64 {
    let n = x.nrows();
    let p = x.ncols();
    let mut means = vec![0.0f64; p];
    for i in 0..n {
        let xi = x.row(i);
        for c in 0..p {
            means[c] += xi[c] as f64;
        }
    }
    for c in 0..p {
        means[c] /= n as f64;
    }
    let mut rss = 0.0f64;
    let mut tss = 0.0f64;
    for i in 0..n {
        let xi = x.row(i);
        let code = &codes[i];
        let recon = reconstruct_stored_code_row(code, decoder, b);
        for c in 0..p {
            let r = xi[c] as f64 - recon[c] as f64;
            rss += r * r;
            let t = xi[c] as f64 - means[c];
            tss += t * t;
        }
    }
    if tss == 0.0 {
        if rss == 0.0 { 1.0 } else { 0.0 }
    } else {
        1.0 - rss / tss
    }
}

fn log_spaced_prefix_atom_counts(n_blocks: usize, b: usize) -> Vec<usize> {
    let mut prefixes = Vec::new();
    let mut blocks = 1usize;
    while blocks < n_blocks {
        prefixes.push(blocks * b);
        blocks = blocks.saturating_mul(2);
    }
    prefixes.push(n_blocks * b);
    prefixes
}

fn prefix_reconstruction_loss(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
    b: usize,
    block_topk: usize,
    minibatch: usize,
    block_tile: usize,
) -> Result<f64, String> {
    let n_blocks = decoder.nrows() / b;
    let k = block_topk.min(n_blocks).max(1);
    let codes = route_and_code_all(x, decoder, gamma, n_blocks, b, k, minibatch, block_tile)?;
    let mut acc = 0.0f64;
    for (i, code) in codes.iter().enumerate() {
        let xi = x.row(i);
        let reconstruction = reconstruct_stored_code_row(code, decoder, b);
        acc += xi
            .iter()
            .zip(reconstruction.iter())
            .map(|(&observed, &fitted)| {
                let residual = observed as f64 - fitted as f64;
                residual * residual
            })
            .sum::<f64>();
    }
    Ok(acc / x.nrows() as f64)
}

fn matryoshka_prefix_losses(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
    n_blocks: usize,
    b: usize,
    block_topk: usize,
    minibatch: usize,
    block_tile: usize,
) -> Result<Vec<(usize, f64)>, String> {
    let mut out = Vec::new();
    let mut best = f64::INFINITY;
    for k_atoms in log_spaced_prefix_atom_counts(n_blocks, b) {
        let prefix_decoder = decoder.slice(ndarray::s![0..k_atoms, ..]);
        let loss = prefix_reconstruction_loss(
            x,
            prefix_decoder,
            gamma,
            b,
            block_topk,
            minibatch,
            block_tile,
        )?;
        best = best.min(loss);
        out.push((k_atoms, best));
    }
    Ok(out)
}

fn matryoshka_block_order(codes: &[RowBlockCode], n_blocks: usize, b: usize) -> Vec<usize> {
    let mut energy = vec![0.0f64; n_blocks];
    for code in codes {
        for (slot, &block) in code.blocks.iter().enumerate() {
            if code.gates[slot] == 0.0 {
                continue;
            }
            let block_index = block as usize;
            for r in 0..b {
                let z = code.codes[slot * b + r] as f64;
                energy[block_index] += z * z;
            }
        }
    }
    let mut order: Vec<usize> = (0..n_blocks).collect();
    order.sort_by(|&left, &right| {
        energy[right]
            .partial_cmp(&energy[left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.cmp(&right))
    });
    order
}

fn reorder_decoder_blocks(decoder: &mut Array2<f32>, order: &[usize], b: usize) {
    let old = decoder.clone();
    for (new_block, &old_block) in order.iter().enumerate() {
        for r in 0..b {
            for c in 0..decoder.ncols() {
                decoder[[new_block * b + r, c]] = old[[old_block * b + r, c]];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reporting: per-block utilisation + stable rank.
// ---------------------------------------------------------------------------

/// Per-block utilisation (fraction of rows selecting each block) and stable rank
/// of the within-block code second moment `C_g = Σ_i z_{ig} z_{ig}ᵀ` (`b×b`):
/// `stable_rank_g = trace(C_g) / λ_max(C_g)`. Both are reported to the MDL lane.
/// Stable rank is a gauge invariant (a similarity `C_g → R C_g Rᵀ` preserves its
/// trace and spectrum), consistent with the block being seen only through its
/// gauge-invariant usage.
fn block_reports(
    codes: &[RowBlockCode],
    n_blocks: usize,
    b: usize,
    n_rows: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut usage = vec![0usize; n_blocks];
    // Per-block b×b second moment.
    let mut second: Vec<Array2<f64>> = (0..n_blocks)
        .map(|_| Array2::<f64>::zeros((b, b)))
        .collect();
    for code in codes.iter() {
        for (j, &g) in code.blocks.iter().enumerate() {
            if code.gates[j] == 0.0 {
                continue;
            }
            let gg = g as usize;
            usage[gg] += 1;
            let z = &code.codes[j * b..j * b + b];
            let cg = &mut second[gg];
            for r1 in 0..b {
                for r2 in 0..b {
                    cg[[r1, r2]] += z[r1] as f64 * z[r2] as f64;
                }
            }
        }
    }
    let util: Vec<f32> = usage
        .iter()
        .map(|&u| u as f32 / n_rows.max(1) as f32)
        .collect();
    let stable: Vec<f32> = second
        .iter()
        .map(|cg| stable_rank_symmetric(cg.view()))
        .collect();
    (util, stable)
}

/// Stable rank `trace(C)/λ_max(C)` of a small symmetric PSD matrix (`b×b`), via a
/// dense symmetric eigensolve. Returns 0 for an all-zero (unused) block.
pub(super) fn stable_rank_symmetric(c: ArrayView2<'_, f64>) -> f32 {
    use gam_linalg::faer_ndarray::FaerEigh;
    let trace: f64 = (0..c.nrows()).map(|i| c[[i, i]]).sum();
    if trace <= 1.0e-24 {
        return 0.0;
    }
    let owned = c.to_owned();
    let lambda_max = match owned.eigh(faer::Side::Lower) {
        Ok((evals, _)) => evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        Err(_) => trace, // degenerate: report rank 1
    };
    if lambda_max <= 1.0e-24 {
        return 0.0;
    }
    (trace / lambda_max) as f32
}

// ---------------------------------------------------------------------------
// Seeding + validation + driver.
// ---------------------------------------------------------------------------

/// Deterministically seed whole subspaces with a block-aware farthest-point
/// pass.  Scalar farthest-point seeding followed by grouping adjacent rows can
/// put directions from unrelated subspaces in the same block.  Every such
/// mixed block may still receive traffic, so dead-block revival cannot repair
/// the resulting live local optimum.
///
/// This is the block analogue of k-means++ / k-subspaces++:
///
/// 1. the next block starts at the row farthest from its nearest completed
///    block projector;
/// 2. the remaining axes maximize uncovered energy times both their affinity
///    to, and novelty beyond, the partial block;
/// 3. nearest-projector residuals are updated after the completed Stiefel frame.
///
/// Orthogonal unrelated subspaces have zero affinity, so they cannot be folded
/// into the same block while a rank-completing row from the anchor subspace is
/// available.  Work is `O(N P G b)`, matching the scalar `K = G b` pass up to
/// the small block factor, and the only corpus-sized scratch is `O(N)` -- no
/// dense `N x K` or second `N x P` object is formed.
pub(super) fn seed_frames(x: ArrayView2<'_, f32>, n_blocks: usize, b: usize) -> Array2<f32> {
    let n = x.nrows();
    let p = x.ncols();
    let row_energy: Vec<f64> = x
        .axis_iter(Axis(0))
        .map(|row| row.iter().map(|&value| (value as f64).powi(2)).sum())
        .collect();
    let mut nearest_projector_residual = row_energy.clone();
    let mut decoder = Array2::<f32>::zeros((n_blocks * b, p));

    for g in 0..n_blocks {
        let anchor = (0..n)
            .max_by(|&left, &right| {
                nearest_projector_residual[left]
                    .total_cmp(&nearest_projector_residual[right])
                    .then_with(|| right.cmp(&left))
            })
            .expect("validated non-empty block dictionary input");

        let mut axes: Vec<Vec<f64>> = Vec::with_capacity(b);
        let mut partial_capture = vec![0.0_f64; n];
        for axis_index in 0..b {
            let row_index = if axis_index == 0 {
                anchor
            } else {
                (0..n)
                    .max_by(|&left, &right| {
                        let score = |row: usize| {
                            let captured = partial_capture[row].min(row_energy[row]);
                            let novel = (row_energy[row] - captured).max(0.0);
                            nearest_projector_residual[row] * captured * novel
                        };
                        score(left)
                            .total_cmp(&score(right))
                            .then_with(|| {
                                nearest_projector_residual[left]
                                    .total_cmp(&nearest_projector_residual[right])
                            })
                            .then_with(|| right.cmp(&left))
                    })
                    .expect("validated non-empty block dictionary input")
            };

            let mut candidate: Vec<f64> =
                x.row(row_index).iter().map(|&value| value as f64).collect();
            let input_norm = row_energy[row_index].sqrt();
            // Two-pass modified Gram--Schmidt removes the component in the
            // partial frame to input precision before the axis is normalized.
            for _ in 0..2 {
                for axis in &axes {
                    let projection: f64 = candidate
                        .iter()
                        .zip(axis.iter())
                        .map(|(left, right)| left * right)
                        .sum();
                    for (value, direction) in candidate.iter_mut().zip(axis.iter()) {
                        *value -= projection * direction;
                    }
                }
            }
            let mut norm = candidate
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let input_roundoff = f32::EPSILON as f64 * (p.max(1) as f64).sqrt() * input_norm;
            if norm <= input_roundoff {
                // The observed rows no longer add rank to this frame.  Complete
                // the required St(b,P) point with the canonical coordinate
                // whose residual against the partial frame is largest.  This
                // is deterministic and threshold-free; p >= b guarantees a
                // positive direction.
                let mut best = vec![0.0_f64; p];
                let mut best_norm2 = f64::NEG_INFINITY;
                for coordinate in 0..p {
                    let mut direction = vec![0.0_f64; p];
                    direction[coordinate] = 1.0;
                    for axis in &axes {
                        let projection: f64 = direction
                            .iter()
                            .zip(axis.iter())
                            .map(|(left, right)| left * right)
                            .sum();
                        for (value, basis_value) in direction.iter_mut().zip(axis.iter()) {
                            *value -= projection * basis_value;
                        }
                    }
                    let norm2 = direction.iter().map(|value| value * value).sum::<f64>();
                    if norm2 > best_norm2 {
                        best_norm2 = norm2;
                        best = direction;
                    }
                }
                norm = best_norm2.sqrt();
                candidate = best;
            }
            for value in &mut candidate {
                *value /= norm;
            }
            axes.push(candidate);

            let axis = axes.last().expect("axis was just installed");
            for row in 0..n {
                let projection: f64 = x
                    .row(row)
                    .iter()
                    .zip(axis.iter())
                    .map(|(&value, direction)| value as f64 * direction)
                    .sum();
                partial_capture[row] += projection * projection;
            }
        }

        let mut block = Array2::<f32>::zeros((b, p));
        for row in 0..b {
            for column in 0..p {
                block[[row, column]] = axes[row][column] as f32;
            }
        }
        orthonormalize_block(&mut block);
        for row in 0..b {
            for column in 0..p {
                decoder[[g * b + row, column]] = block[[row, column]];
            }
        }

        for row in 0..n {
            let mut captured = 0.0_f64;
            for axis in 0..b {
                let projection: f64 = x
                    .row(row)
                    .iter()
                    .zip(block.row(axis).iter())
                    .map(|(&value, &direction)| value as f64 * direction as f64)
                    .sum();
                captured += projection * projection;
            }
            let residual = (row_energy[row] - captured).max(0.0);
            nearest_projector_residual[row] = nearest_projector_residual[row].min(residual);
        }
    }
    decoder
}

/// Relative Grassmann-projector displacement between two block dictionaries.
/// For each block this evaluates `||DᵀD-EᵀE||_F` from only `b×b` frame
/// overlaps. The expression uses the measured projector norms rather than
/// assuming exact floating-point orthonormality, so identical stored frames have
/// exactly zero residual. Every term is invariant to independent `O(b)` changes
/// of basis in either frame.
fn frame_fixed_point_residual(
    previous: ArrayView2<'_, f32>,
    next: ArrayView2<'_, f32>,
    n_blocks: usize,
    b: usize,
) -> f64 {
    let mut maximum = 0.0_f64;
    for block in 0..n_blocks {
        let mut previous_norm2 = 0.0_f64;
        let mut next_norm2 = 0.0_f64;
        let mut overlap = 0.0_f64;
        for left_axis in 0..b {
            for right_axis in 0..b {
                let mut previous_dot = 0.0_f64;
                let mut next_dot = 0.0_f64;
                let mut cross_dot = 0.0_f64;
                for column in 0..previous.ncols() {
                    previous_dot += previous[[block * b + left_axis, column]] as f64
                        * previous[[block * b + right_axis, column]] as f64;
                    next_dot += next[[block * b + left_axis, column]] as f64
                        * next[[block * b + right_axis, column]] as f64;
                    cross_dot += previous[[block * b + left_axis, column]] as f64
                        * next[[block * b + right_axis, column]] as f64;
                }
                previous_norm2 += previous_dot * previous_dot;
                next_norm2 += next_dot * next_dot;
                overlap += cross_dot * cross_dot;
            }
        }
        let scale = previous_norm2 + next_norm2;
        let distance2 = (scale - 2.0 * overlap).max(0.0);
        let residual = if scale == 0.0 {
            if distance2 == 0.0 { 0.0 } else { f64::INFINITY }
        } else {
            (distance2 / scale).sqrt()
        };
        maximum = maximum.max(residual);
    }
    maximum
}

fn relative_scalar_change(previous: f32, current: f32) -> f64 {
    let previous = previous as f64;
    let current = current as f64;
    (current - previous).abs() / previous.abs().max(current.abs()).max(f64::MIN_POSITIVE)
}

#[derive(Clone)]
struct BlockSparseState {
    decoder: Array2<f32>,
    codes: Vec<RowBlockCode>,
    gamma: f32,
    explained_variance: f64,
}

struct BlockSparseStep {
    next: BlockSparseState,
    accepted_births: usize,
    polar_failures: usize,
}

fn stored_code_gate(code: &RowBlockCode, slot: usize, b: usize) -> f64 {
    code.codes[slot * b..slot * b + b]
        .iter()
        .map(|&value| {
            let value = value as f64;
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

fn gate_for_block(code: &RowBlockCode, block: u32, b: usize) -> f64 {
    code.blocks
        .iter()
        .enumerate()
        .filter(|&(slot, candidate)| *candidate == block && code.gates[slot] != 0.0)
        .map(|(slot, _)| stored_code_gate(code, slot, b))
        .sum()
}

/// Gauge-invariant fixed-point residuals for the exposed sparse routing and its
/// reconstruction. Routing compares the `l2` norm of each selected block code;
/// reconstruction compares the actual stored-code decodes without allocating an
/// `N×P` matrix. Both residuals are relative squared displacements, matching the
/// scale-free tolerance used by the atom dictionary lane.
fn routing_and_reconstruction_residuals(
    x: ArrayView2<'_, f32>,
    previous: &BlockSparseState,
    next: &BlockSparseState,
    b: usize,
) -> (f64, f64) {
    let mut gate_delta2 = 0.0_f64;
    let mut gate_scale2 = 0.0_f64;
    let mut reconstruction_delta2 = 0.0_f64;
    let mut data_scale2 = 0.0_f64;

    for row in 0..x.nrows() {
        let old_code = &previous.codes[row];
        let new_code = &next.codes[row];
        for (slot, &block) in old_code.blocks.iter().enumerate() {
            if old_code.gates[slot] == 0.0 {
                continue;
            }
            let old_gate = stored_code_gate(old_code, slot, b);
            let new_gate = gate_for_block(new_code, block, b);
            let delta = new_gate - old_gate;
            gate_delta2 += delta * delta;
            gate_scale2 += old_gate * old_gate + new_gate * new_gate;
        }
        for (slot, &block) in new_code.blocks.iter().enumerate() {
            if new_code.gates[slot] == 0.0
                || old_code
                    .blocks
                    .iter()
                    .enumerate()
                    .any(|(old_slot, candidate)| {
                        *candidate == block && old_code.gates[old_slot] != 0.0
                    })
            {
                continue;
            }
            let new_gate = stored_code_gate(new_code, slot, b);
            gate_delta2 += new_gate * new_gate;
            gate_scale2 += new_gate * new_gate;
        }

        let old_reconstruction = reconstruct_stored_code_row(old_code, previous.decoder.view(), b);
        let new_reconstruction = reconstruct_stored_code_row(new_code, next.decoder.view(), b);
        for column in 0..x.ncols() {
            let delta = new_reconstruction[column] as f64 - old_reconstruction[column] as f64;
            reconstruction_delta2 += delta * delta;
            let observed = x[[row, column]] as f64;
            data_scale2 += observed * observed;
        }
    }

    let routing_residual = if gate_scale2 == 0.0 {
        if gate_delta2 == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        gate_delta2 / gate_scale2
    };
    let reconstruction_residual = if data_scale2 == 0.0 {
        if reconstruction_delta2 == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        reconstruction_delta2 / data_scale2
    };
    (routing_residual, reconstruction_residual)
}

fn route_and_close_gamma(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    gamma_seed: f32,
    config: &BlockSparseConfig,
    k: usize,
) -> Result<(f32, Vec<RowBlockCode>), String> {
    let routed = route_and_code_all(
        x,
        decoder,
        gamma_seed,
        config.n_blocks,
        config.block_size,
        k,
        config.minibatch,
        config.block_tile,
    )?;
    let gamma = refresh_gamma(x, &routed, decoder, config.block_size);
    if !gamma.is_finite() || gamma < 0.0 {
        return Err(format!(
            "fit_block_sparse_dictionary gamma refresh produced invalid scale {gamma}"
        ));
    }
    let codes = route_and_code_all(
        x,
        decoder,
        gamma,
        config.n_blocks,
        config.block_size,
        k,
        config.minibatch,
        config.block_tile,
    )?;
    Ok((gamma, codes))
}

/// Put a state into the deterministic MATRYOSHKA block labelling before it can
/// be certified. Decoder rows and stored sparse indices are permuted together,
/// so this is an exact representation change: codes, gates, gamma, routing, and
/// reconstruction do not need to be recomputed afterward.
fn canonicalize_matryoshka_state(state: &mut BlockSparseState, config: &BlockSparseConfig) {
    if !config.matryoshka_prefix {
        return;
    }
    let order = matryoshka_block_order(&state.codes, config.n_blocks, config.block_size);
    if order
        .iter()
        .enumerate()
        .all(|(index, &block)| index == block)
    {
        return;
    }
    reorder_decoder_blocks(&mut state.decoder, &order, config.block_size);
    let mut old_to_new = vec![0usize; config.n_blocks];
    for (new_block, &old_block) in order.iter().enumerate() {
        old_to_new[old_block] = new_block;
    }
    for code in &mut state.codes {
        for (slot, block) in code.blocks.iter_mut().enumerate() {
            if code.gates[slot] != 0.0 {
                *block = old_to_new[*block as usize] as u32;
            }
        }
    }
}

fn proposal_is_selected(codes: &[RowBlockCode], block: usize, b: usize) -> bool {
    codes.iter().any(|code| {
        code.blocks.iter().enumerate().any(|(slot, &candidate)| {
            candidate as usize == block
                && code.gates[slot] != 0.0
                && stored_code_gate(code, slot, b) > 0.0
        })
    })
}

/// Replay one complete deterministic alternation from `current`. The caller
/// compares `current` with the returned image and, on convergence, returns
/// `current`: the model API therefore exposes exactly the state whose full-step
/// residual was measured.
fn advance_block_sparse_state(
    x: ArrayView2<'_, f32>,
    current: &BlockSparseState,
    config: &BlockSparseConfig,
    k: usize,
) -> Result<BlockSparseStep, String> {
    let b = config.block_size;
    let gamma_for_refresh = refresh_gamma(x, &current.codes, current.decoder.view(), b);
    if !gamma_for_refresh.is_finite() || gamma_for_refresh < 0.0 {
        return Err(format!(
            "fit_block_sparse_dictionary gamma refresh produced invalid scale {gamma_for_refresh}"
        ));
    }
    let codes_for_refresh = route_and_code_all(
        x,
        current.decoder.view(),
        gamma_for_refresh,
        config.n_blocks,
        b,
        k,
        config.minibatch,
        config.block_tile,
    )?;

    let mut decoder = current.decoder.clone();
    let polar_failures = refresh_frames(
        x,
        &codes_for_refresh,
        &mut decoder,
        config.n_blocks,
        b,
        config.frame_ridge,
    );
    let proposals = revive_dead_blocks(
        x,
        &codes_for_refresh,
        &mut decoder,
        config.n_blocks,
        b,
        config.aux_k,
    );

    // A residual-row frame is a proposal, not a model parameter. Re-route after
    // installing all proposals, restore every proposal that did not receive a
    // nonzero tied code, and repeat because one restoration can displace another
    // marginal proposal. At least one proposal is removed on every repeat.
    let mut retained = vec![true; proposals.len()];
    let mut gamma_seed = gamma_for_refresh;
    let (gamma, codes) = loop {
        let (candidate_gamma, candidate_codes) =
            route_and_close_gamma(x, decoder.view(), gamma_seed, config, k)?;
        let rejected: Vec<usize> = proposals
            .iter()
            .enumerate()
            .filter(|(proposal_index, proposal)| {
                retained[*proposal_index]
                    && !proposal_is_selected(&candidate_codes, proposal.block, b)
            })
            .map(|(proposal_index, _)| proposal_index)
            .collect();
        if rejected.is_empty() {
            break (candidate_gamma, candidate_codes);
        }
        for proposal_index in rejected {
            let proposal = &proposals[proposal_index];
            for row in 0..b {
                for column in 0..decoder.ncols() {
                    decoder[[proposal.block * b + row, column]] =
                        proposal.previous_frame[[row, column]];
                }
            }
            retained[proposal_index] = false;
        }
        gamma_seed = candidate_gamma;
    };
    let accepted_births = retained.into_iter().filter(|&accepted| accepted).count();
    let explained_variance = explained_variance(x, &codes, decoder.view(), b);
    if !explained_variance.is_finite() {
        return Err("fit_block_sparse_dictionary produced non-finite explained variance".into());
    }
    let mut next = BlockSparseState {
        decoder,
        codes,
        gamma,
        explained_variance,
    };
    canonicalize_matryoshka_state(&mut next, config);

    Ok(BlockSparseStep {
        next,
        accepted_births,
        polar_failures,
    })
}

fn validate(x: ArrayView2<'_, f32>, config: &BlockSparseConfig) -> Result<(), BlockSparseFitError> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err(BlockSparseFitError::invalid_input(
            "fit_block_sparse_dictionary requires a non-empty N×P matrix",
        ));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(BlockSparseFitError::invalid_input(
            "fit_block_sparse_dictionary input must be finite",
        ));
    }
    if config.n_blocks == 0 {
        return Err(BlockSparseFitError::invalid_input(
            "fit_block_sparse_dictionary requires n_blocks >= 1",
        ));
    }
    if config.block_size == 0 {
        return Err(BlockSparseFitError::invalid_input(
            "fit_block_sparse_dictionary requires block_size >= 1",
        ));
    }
    if config.block_size > x.ncols() {
        return Err(BlockSparseFitError::invalid_input(format!(
            "fit_block_sparse_dictionary block_size b={} cannot exceed output dim P={} \
             (a block's b orthonormal rows must fit in ℝ^P)",
            config.block_size,
            x.ncols()
        )));
    }
    if config.block_topk == 0 {
        return Err(BlockSparseFitError::invalid_input(
            "fit_block_sparse_dictionary requires block_topk >= 1",
        ));
    }
    if config.block_topk > config.n_blocks {
        return Err(BlockSparseFitError::invalid_input(format!(
            "fit_block_sparse_dictionary block_topk={} exceeds n_blocks={}; the active budget is never clamped",
            config.block_topk, config.n_blocks
        )));
    }
    if config.max_epochs == 0 {
        return Err(BlockSparseFitError::invalid_input(
            "fit_block_sparse_dictionary requires max_epochs >= 1",
        ));
    }
    if !(config.frame_ridge.is_finite() && config.frame_ridge >= 0.0) {
        return Err(BlockSparseFitError::invalid_input(
            "fit_block_sparse_dictionary frame_ridge must be finite and >= 0",
        ));
    }
    if !(config.tolerance.is_finite() && config.tolerance >= 0.0) {
        return Err(BlockSparseFitError::invalid_input(
            "fit_block_sparse_dictionary tolerance must be finite and non-negative",
        ));
    }
    Ok(())
}

/// Fit a block-sparse dictionary to `x` (`N×P`): `G` blocks of `b` orthonormal
/// atoms, block-TopK routing by group ℓ₂ gate, tied signed codes with one shared
/// scalar `γ`, Stiefel-constrained frames refreshed by polar steps, and AuxK
/// dead-block revival. Never forms a dense `N×K` object.
///
/// # One-engine position (design gam#2232, Increment 5b)
///
/// A block IS a framed Euclidean `d = b` atom of the one engine: decoder
/// `B_g = C_g·U_gᵀ` with `U_g ∈ Gr(b, P)` (the same profiled-frame
/// representation the curved engine's `decoder_frame` / `factored_border_dim`
/// carry), block-TopK = hard top-k support at atom granularity, and the
/// within-block code `z_g` = the atom's Euclidean latent coordinate. This
/// alternation — projection code solve on orthonormal frames, polar frame
/// refresh — is the BLOCK FAST KERNEL of that model (the `d = b` sibling of
/// [`super::update::run_linear_fast_kernel`]): the code solve is the exact
/// degenerate arrow-Schur inner solve for read-only gates on linear atoms
/// (projection, no ridge pair to unify), and the polar step is the Grassmann
/// retraction of the framed decoder refresh. The single public entry
/// (`sae_manifold_fit`, `atom_topology="linear"` + uniform `d_atom = b ≥ 2` +
/// hard top-k) reaches this kernel through
/// [`crate::front_door::admit_linear_dictionary`] — an explicit
/// linear-dictionary request is a modeling choice admitted at ANY `K`, not a
/// shape-derived demotion.
pub fn fit_block_sparse_dictionary(
    x: ArrayView2<'_, f32>,
    config: &BlockSparseConfig,
) -> Result<BlockSparseFit, BlockSparseFitError> {
    validate(x, config)?;
    let n = x.nrows();
    let g = config.n_blocks;
    let b = config.block_size;
    let k = config.block_topk.min(g).max(1);

    let decoder = seed_frames(x, g, b);
    let gamma = 1.0f32;
    let codes = route_and_code_all(
        x,
        decoder.view(),
        gamma,
        g,
        b,
        k,
        config.minibatch,
        config.block_tile,
    )?;
    let seed_ev = explained_variance(x, &codes, decoder.view(), b);
    let mut state = BlockSparseState {
        decoder,
        codes,
        gamma,
        explained_variance: seed_ev,
    };
    canonicalize_matryoshka_state(&mut state, config);

    let mut converged = false;
    let mut epochs_run = 0usize;
    let mut ev_residual = f64::INFINITY;
    let mut gamma_residual = f64::INFINITY;
    let mut frame_residual = f64::INFINITY;
    let routing_residual: f64;
    let reconstruction_residual: f64;
    let mut accepted_births = 0usize;
    let mut polar_failures = 0usize;

    for epoch in 0..config.max_epochs {
        epochs_run = epoch + 1;
        let step = advance_block_sparse_state(x, &state, config, k)?;
        ev_residual = relative_scalar_change(
            state.explained_variance as f32,
            step.next.explained_variance as f32,
        );
        gamma_residual = relative_scalar_change(state.gamma, step.next.gamma);
        frame_residual =
            frame_fixed_point_residual(state.decoder.view(), step.next.decoder.view(), g, b);
        accepted_births = step.accepted_births;
        polar_failures = step.polar_failures;
        state = step.next;

        let fixed_point_residual = ev_residual.max(gamma_residual).max(frame_residual);
        if accepted_births == 0
            && polar_failures == 0
            && fixed_point_residual <= config.tolerance
            && epoch > 0
        {
            converged = true;
            break;
        }
    }

    // Certificate replay: ONE more full alternation from the (candidate)
    // fixed point, recording the gauge-invariant routing / reconstruction
    // displacements it causes. This is the "fixed-point evidence" the
    // convergence struct documents — measured by replay, never used as the
    // per-epoch stop rule (a strict per-epoch routing gate at high `K` churns
    // near degeneracy and can never clear a 1e-6 tolerance).
    {
        let replay = advance_block_sparse_state(x, &state, config, k)?;
        let (routing, reconstruction) =
            routing_and_reconstruction_residuals(x, &state, &replay.next, b);
        routing_residual = routing;
        reconstruction_residual = reconstruction;
        if converged && (replay.accepted_births != 0 || replay.polar_failures != 0) {
            // A replay that still births or fails polar subsolves is not a
            // certified fixed point — surface it as the evidence values.
            accepted_births = replay.accepted_births;
            polar_failures = replay.polar_failures;
            converged = false;
        }
    }

    if !converged {
        return Err(BlockSparseFitError::NonConvergence {
            epochs: epochs_run,
            explained_variance: state.explained_variance,
            ev_residual,
            gamma_residual,
            frame_residual,
            routing_residual,
            reconstruction_residual,
            tolerance: config.tolerance,
            accepted_births,
            polar_failures,
        });
    }

    let BlockSparseState {
        mut decoder,
        mut codes,
        gamma,
        explained_variance: _,
    } = state;

    // One final γ refresh against the last routing so the returned scalar is the
    // exact least-squares fit to the returned frames + codes.
    let gamma_prev = gamma;
    let gamma = refresh_gamma(x, &codes, decoder.view(), b);
    if config.matryoshka_prefix {
        let order = matryoshka_block_order(&codes, g, b);
        reorder_decoder_blocks(&mut decoder, &order, b);
        codes = route_and_code_all(
            x,
            decoder.view(),
            gamma,
            g,
            b,
            k,
            config.minibatch,
            config.block_tile,
        )?;
    } else if gamma_prev > 0.0 && gamma != gamma_prev {
        // The signed code is LINEAR in γ (`z_g = γ·w_g`) and the routing order
        // is γ-invariant, so rescaling the last encode by `γ_new/γ_old` IS the
        // exact re-encode under the final γ. Without it the packed codes stay
        // at the pre-refresh scale while `gates`, `gamma`, and the EV below use
        // the refreshed one — the artifact would violate its own invariants
        // (`reconstruct()` disagreeing with `explained_variance`, and
        // `gate ≠ ‖z_g‖₂`).
        let rescale = gamma / gamma_prev;
        for code in codes.iter_mut() {
            for z in code.codes.iter_mut() {
                *z *= rescale;
            }
            for gate in code.gates.iter_mut() {
                *gate *= rescale;
            }
        }
    }
    let final_ev = explained_variance(x, &codes, decoder.view(), b);
    let (block_utilization, block_stable_rank) = block_reports(&codes, g, b, n);
    let prefix_losses = if config.matryoshka_prefix {
        matryoshka_prefix_losses(
            x,
            decoder.view(),
            gamma,
            g,
            b,
            k,
            config.minibatch,
            config.block_tile,
        )?
    } else {
        Vec::new()
    };

    // Pack the fixed-width sparse routing. The gate is recomputed as the group
    // ℓ₂ `‖z_g‖₂ = γ·‖x D_gᵀ‖₂` under the FINAL γ + frames (the codes were last
    // encoded before the final γ refresh); the signed within-block codes come
    // straight from that last encode.
    let mut blocks = Array2::<u32>::zeros((n, k));
    let mut gates = Array2::<f32>::zeros((n, k));
    let mut code_arr = Array3::<f32>::zeros((n, k, b));
    for (i, code) in codes.iter().enumerate() {
        for j in 0..k {
            blocks[[i, j]] = code.blocks[j];
            for r in 0..b {
                code_arr[[i, j, r]] = code.codes[j * b + r];
            }
        }
    }
    recompute_gates(x, decoder.view(), &blocks, gamma, b, &mut gates);

    Ok(BlockSparseFit {
        decoder,
        blocks,
        gates,
        codes: code_arr,
        gamma,
        block_utilization,
        block_stable_rank,
        matryoshka_prefix_losses: prefix_losses,
        explained_variance: final_ev,
        epochs: epochs_run,
        convergence: BlockSparseConvergence {
            ev_residual,
            gamma_residual,
            frame_residual,
            routing_residual,
            reconstruction_residual,
            accepted_births,
            polar_failures,
            tolerance: config.tolerance,
        },
        block_topk: k,
        block_size: b,
    })
}

/// Overwrite `gates[i,j] = γ·‖x_i D_{g}ᵀ‖₂` for the packed routing, so the stored
/// gate is exactly the presence signal `‖z_g‖₂` under the FINAL `γ` and frames
/// (the codes were last encoded before the final γ refresh; the gate is defined
/// as the group ℓ₂ of the current signed code).
fn recompute_gates(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: &Array2<u32>,
    gamma: f32,
    b: usize,
    gates: &mut Array2<f32>,
) {
    let (n, k) = blocks.dim();
    for i in 0..n {
        let xi = x.row(i);
        for j in 0..k {
            let g = blocks[[i, j]] as usize;
            let mut e = 0.0f32;
            for r in 0..b {
                let atom = decoder.row(g * b + r);
                let mut wr = 0.0f32;
                for (xr, ar) in xi.iter().zip(atom.iter()) {
                    wr += *xr * *ar;
                }
                e += wr * wr;
            }
            // Padded slots (block 0 with zero true gate) resolve to 0 only when the
            // projection is genuinely zero; a real block-0 selection keeps its gate.
            gates[[i, j]] = gamma.abs() * e.sqrt();
        }
    }
}

/// Out-of-sample block encode: route held-out rows `x` (`M×P`) against frozen
/// block frames `decoder` (`K×P`, `K = G·b`) with tied scalar `gamma`, returning
/// the fixed-width sparse block routing `(blocks[M,k], gates[M,k], codes[M,k,b])`.
///
/// This is the Rust core of the block lane's `transform`: the same group-ℓ₂ gate
/// (`‖z_g‖₂ = γ‖x D_gᵀ‖₂`), block-TopK selection, and tied signed within-block
/// codes (`z_g = γ x D_gᵀ`, no ReLU) the trainer uses — so held-out encoding is
/// bit-consistent with training, and the Python facade need not reimplement it in
/// numpy. `gates` carry the FINAL-γ presence `γ·‖x D_gᵀ‖₂`; `codes` are the signed
/// `z_g`.
pub fn block_sparse_dictionary_transform(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
    block_size: usize,
    block_topk: usize,
    block_tile: usize,
) -> Result<(Array2<u32>, Array2<f32>, Array3<f32>), String> {
    let b = block_size;
    if b == 0 {
        return Err("block_sparse_dictionary_transform: block_size must be >= 1".to_string());
    }
    let krows = decoder.nrows();
    if krows == 0 || krows % b != 0 {
        return Err(format!(
            "block_sparse_dictionary_transform: decoder has K={krows} rows, not a multiple of \
             block_size b={b}"
        ));
    }
    if x.ncols() != decoder.ncols() {
        return Err(format!(
            "block_sparse_dictionary_transform: X has P={} columns but the frames have P={}",
            x.ncols(),
            decoder.ncols()
        ));
    }
    let g = krows / b;
    let k = block_topk.min(g).max(1);
    // Route + tied-code every row (block-tiled internally, never N×K). A generous
    // minibatch keeps the peak working set bounded without materialising M×G.
    let minibatch = 4096usize;
    let codes = route_and_code_all(x, decoder, gamma, g, b, k, minibatch, block_tile.max(1))?;

    let m = x.nrows();
    let mut blocks = Array2::<u32>::zeros((m, k));
    let mut gates = Array2::<f32>::zeros((m, k));
    let mut code_arr = Array3::<f32>::zeros((m, k, b));
    for (i, code) in codes.iter().enumerate() {
        for j in 0..k {
            blocks[[i, j]] = code.blocks[j];
            // Presence gate under the tied scalar: γ·‖w_g‖₂ = ‖z_g‖₂.
            gates[[i, j]] = gamma.abs() * code.gates[j];
            for r in 0..b {
                code_arr[[i, j, r]] = code.codes[j * b + r];
            }
        }
    }
    Ok((blocks, gates, code_arr))
}

/// Dense reconstruction from fixed-width block routing.
pub fn reconstruct_block_sparse_rows(
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    block_size: usize,
) -> Result<Array2<f32>, String> {
    let b = block_size;
    if b == 0 {
        return Err("reconstruct_block_sparse_rows: block_size must be >= 1".to_string());
    }
    if decoder.nrows() % b != 0 {
        return Err(format!(
            "reconstruct_block_sparse_rows: decoder rows {} not divisible by block_size {b}",
            decoder.nrows()
        ));
    }
    let (n, k) = blocks.dim();
    if codes.shape() != [n, k, b] {
        return Err(format!(
            "reconstruct_block_sparse_rows: codes shape {:?} does not match ({n}, {k}, {b})",
            codes.shape()
        ));
    }
    let g = decoder.nrows() / b;
    let p = decoder.ncols();
    let mut out = Array2::<f32>::zeros((n, p));
    for i in 0..n {
        for j in 0..k {
            let block = blocks[[i, j]] as usize;
            if block >= g {
                return Err(format!(
                    "reconstruct_block_sparse_rows: block index {block} out of range 0..{g}"
                ));
            }
            for r in 0..b {
                let code = codes[[i, j, r]];
                if code == 0.0 {
                    continue;
                }
                let atom = decoder.row(block * b + r);
                for c in 0..p {
                    out[[i, c]] += code * atom[c];
                }
            }
        }
    }
    Ok(out)
}

/// Project rows into one block frame: `X D_g^T`.
pub fn block_sparse_dictionary_block_coords(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    block_size: usize,
    block: usize,
) -> Result<Array2<f32>, String> {
    let b = block_size;
    if b == 0 {
        return Err("block_sparse_dictionary_block_coords: block_size must be >= 1".to_string());
    }
    if decoder.nrows() % b != 0 {
        return Err(format!(
            "block_sparse_dictionary_block_coords: decoder rows {} not divisible by block_size {b}",
            decoder.nrows()
        ));
    }
    if x.ncols() != decoder.ncols() {
        return Err(format!(
            "block_sparse_dictionary_block_coords: X has P={} columns but decoder has P={}",
            x.ncols(),
            decoder.ncols()
        ));
    }
    let g = decoder.nrows() / b;
    if block >= g {
        return Err(format!(
            "block_sparse_dictionary_block_coords: block {block} out of range 0..{g}"
        ));
    }
    let n = x.nrows();
    let p = x.ncols();
    let mut out = Array2::<f32>::zeros((n, b));
    for i in 0..n {
        for r in 0..b {
            let atom = decoder.row(block * b + r);
            let mut dot = 0.0f32;
            for c in 0..p {
                dot += x[[i, c]] * atom[c];
            }
            out[[i, r]] = dot;
        }
    }
    Ok(out)
}

/// Lift block coordinates to ambient rows: `coords D_g`.
pub fn block_sparse_dictionary_lift_block(
    coords: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    block_size: usize,
    block: usize,
) -> Result<Array2<f32>, String> {
    let b = block_size;
    if b == 0 {
        return Err("block_sparse_dictionary_lift_block: block_size must be >= 1".to_string());
    }
    if coords.ncols() != b {
        return Err(format!(
            "block_sparse_dictionary_lift_block: coords has {} columns, expected block_size {b}",
            coords.ncols()
        ));
    }
    if decoder.nrows() % b != 0 {
        return Err(format!(
            "block_sparse_dictionary_lift_block: decoder rows {} not divisible by block_size {b}",
            decoder.nrows()
        ));
    }
    let g = decoder.nrows() / b;
    if block >= g {
        return Err(format!(
            "block_sparse_dictionary_lift_block: block {block} out of range 0..{g}"
        ));
    }
    let n = coords.nrows();
    let p = decoder.ncols();
    let mut out = Array2::<f32>::zeros((n, p));
    for i in 0..n {
        for r in 0..b {
            let code = coords[[i, r]];
            if code == 0.0 {
                continue;
            }
            let atom = decoder.row(block * b + r);
            for c in 0..p {
                out[[i, c]] += code * atom[c];
            }
        }
    }
    Ok(out)
}

/// Leave-one-block-out residual target projected into block coordinates.
/// [`block_sparse_dictionary_project_residual`] with the leave-one-block-out
/// base taken from CALLER-SUPPLIED codes instead of a fresh tied transform.
/// The co-fit alternation refits the linear codes between chart passes; a chart
/// subproblem for block `g` must then be solved against `x − L_{−g}(codes)`
/// built from THOSE codes — re-deriving tied codes from `x` here would hand the
/// chart a residual belonging to a different (stale) linear tier, so the block
/// coordinate-descent would not be minimizing its stated objective.
pub fn block_sparse_dictionary_project_residual_with_codes(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    blocks: ArrayView2<'_, u32>,
    codes: ArrayView3<'_, f32>,
    block_size: usize,
    block: usize,
) -> Result<Array2<f32>, String> {
    let xhat = reconstruct_block_sparse_rows(decoder, blocks, codes, block_size)?;
    let mut residual = x.to_owned();
    residual -= &xhat;
    let b = block_size;
    for i in 0..x.nrows() {
        for j in 0..blocks.ncols() {
            if blocks[[i, j]] as usize != block {
                continue;
            }
            for r in 0..b {
                let code = codes[[i, j, r]];
                if code == 0.0 {
                    continue;
                }
                let atom = decoder.row(block * b + r);
                for c in 0..x.ncols() {
                    residual[[i, c]] += code * atom[c];
                }
            }
        }
    }
    block_sparse_dictionary_block_coords(residual.view(), decoder, block_size, block)
}

pub fn block_sparse_dictionary_project_residual(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
    block_size: usize,
    block_topk: usize,
    block_tile: usize,
    block: usize,
) -> Result<Array2<f32>, String> {
    let (blocks, _gates, codes) =
        block_sparse_dictionary_transform(x, decoder, gamma, block_size, block_topk, block_tile)?;
    let xhat = reconstruct_block_sparse_rows(decoder, blocks.view(), codes.view(), block_size)?;
    let mut residual = x.to_owned();
    residual -= &xhat;
    let b = block_size;
    for i in 0..x.nrows() {
        for j in 0..blocks.ncols() {
            if blocks[[i, j]] as usize != block {
                continue;
            }
            for r in 0..b {
                let code = codes[[i, j, r]];
                if code == 0.0 {
                    continue;
                }
                let atom = decoder.row(block * b + r);
                for c in 0..x.ncols() {
                    residual[[i, c]] += code * atom[c];
                }
            }
        }
    }
    block_sparse_dictionary_block_coords(residual.view(), decoder, block_size, block)
}

#[cfg(test)]
#[path = "block_tests.rs"]
mod block_tests;
