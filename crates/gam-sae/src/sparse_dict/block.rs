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

    /// Dictionary width `K = G·b`.
    pub fn n_atoms(&self) -> usize {
        self.n_blocks * self.block_size
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
    /// block's `b` rows are orthonormal (`D_g D_gᵀ = I_b`).
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
    /// Whether the EV-improvement tolerance was reached.
    pub converged: bool,
    /// Block budget `k` actually used (`min(block_topk, G)`).
    pub block_topk: usize,
    /// Block size `b` actually used.
    pub block_size: usize,
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
fn route_block_minibatch(
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

/// One-shot bounded-coverage note: a device-present host declined a below-break-even
/// block minibatch and ran the exact CPU oracle instead. Routed through `log::warn!`
/// (the repo's sanctioned diagnostics path) and fired once per process so the
/// per-minibatch route does not spam thousands of identical lines. This is the "log
/// what is dropped, no silent cap" record for the small-`K` CPU baseline path.
#[cfg(target_os = "linux")]
fn note_below_breakeven_cpu_route(n_rows: usize, krows: usize) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        log::warn!(
            "[gam-sae sparse_dict block-gate router] below-break-even CPU route: block \
             {n_rows}x{krows} = {} elems is under the device launch break-even \
             (DEVICE_BLOCK_GATE_MIN_ELEMS={}); running the exact CPU oracle (device could \
             not beat it) — this is the small-K CPU baseline, not a silent GPU-0% fallback",
            n_rows.saturating_mul(krows),
            gam_gpu::DEFAULT_DICTIONARY_SCORE_MIN_ELEMS,
        );
    });
}

/// Route one minibatch's blocks, dispatching to the CUDA block-gate router when a
/// device residency mode asks for it and a CUDA runtime is actually present, and
/// to the CPU router ([`route_block_minibatch`]) otherwise.
///
/// The dispatch honours the process-wide [`gam_gpu::GpuMode`]: `Off` (or any
/// device-absent host, which is all of CI) always takes the exact CPU router, so
/// device-absent behaviour is bit-for-bit unchanged; `Auto` uses the device when
/// admitted and above break-even and transparently falls back to the CPU oracle
/// otherwise (the fallback is logged once by the block-gate router, never
/// silent); `Required` propagates an ADMITTED (above-break-even) device fault as a
/// typed `Err` up through the fallible fit rather than degrading silently. The
/// device route carries the #2227 bounded-progress checkpoints, so a device stall
/// surfaces as a tile-attributed error instead of a silent hang.
///
/// BELOW-BREAK-EVEN (small-`K`) blocks take the exact CPU oracle under EVERY mode,
/// including `Required` (#2134): below the `n_rows × K` launch break-even the device
/// provably cannot beat the CPU, so refusing there would guard nothing (there is no
/// silent GPU-0% regime to catch) while making a small-`K` CPU baseline impossible
/// under a `Required` process. The bounded CPU coverage is logged once, never a
/// silent cap.
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
        let mode = gam_gpu::gpu_mode();
        if mode != gam_gpu::GpuMode::Off && gam_gpu::GpuRuntime::global().is_some() {
            // Only hand a minibatch to the device when the `n_rows × K` score work
            // clears the launch break-even. Below it the device provably cannot beat
            // the CPU (that IS the definition of the break-even), so a `Required`
            // refusal there guards NOTHING — there is no #1551 "GPU 0%" regime to
            // catch when the device could not have helped in the first place. We
            // take the exact CPU router (`route_block_minibatch`) and log the bounded
            // coverage once, so a small-`K` block fit yields a CPU baseline under any
            // residency mode. The fail-loud `Required` contract is preserved exactly
            // where it matters: for an ADMITTED (above-break-even) block a device
            // fault still propagates as a typed `Err` through `route_blocks_required`.
            let admitted = gam_gpu::DictionaryScoreRoutePlan::default_for_shape(
                mb.nrows(),
                decoder.nrows(),
                decoder.ncols(),
            )
            .device_admitted;
            if admitted {
                let (selections, _path, _dtoh) =
                    super::block_scoring_gpu::route_blocks_required(mb, decoder, b, k, mode)
                        .map_err(|err| err.to_string())?;
                return Ok(selections);
            }
            note_below_breakeven_cpu_route(mb.nrows(), decoder.nrows());
        }
    }
    Ok(route_block_minibatch(mb, decoder, n_blocks, b, k, block_tile))
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

/// Closed-form refresh of the shared tied scalar
/// `γ* = (Σ_i ⟨x_i, p_i⟩) / (Σ_i ‖p_i‖²)`, where `p_i = Σ_{g∈S_i} x_i P_g`. This is
/// the exact least-squares `γ` given the current frames and routing (decode is
/// `γ p_i`). Returns the previous `γ` if the denominator underflows (no block
/// fired anywhere).
fn refresh_gamma(
    x: ArrayView2<'_, f32>,
    codes: &[RowBlockCode],
    decoder: ArrayView2<'_, f32>,
    b: usize,
    prev_gamma: f32,
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
    if den <= 1.0e-24 {
        prev_gamma
    } else {
        (num / den) as f32
    }
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
    gamma: f32,
    n_blocks: usize,
    b: usize,
    ridge: f64,
) {
    let p = x.ncols();
    // Cross-moments M_g (P×b), one per block.
    let mut cm: Vec<Array2<f64>> = (0..n_blocks)
        .map(|_| Array2::<f64>::zeros((p, b)))
        .collect();
    let mut touched = vec![false; n_blocks];

    for (i, code) in codes.iter().enumerate() {
        let xi = x.row(i);
        // Full reconstruction x̂_i under the current frames/γ.
        let selected: Vec<u32> = code
            .blocks
            .iter()
            .zip(code.gates.iter())
            .filter(|&(_, &gate)| gate != 0.0)
            .map(|(&g, _)| g)
            .collect();
        if selected.is_empty() {
            continue;
        }
        let recon = reconstruct_row(xi, decoder.view(), &selected, gamma, b);
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
        if let Ok(frame) = GrassmannFrame::polar_update(cm[g].view()) {
            let u = frame.frame(); // P×b column-orthonormal
            let sv = frame.gauge_singular_values();
            let full_rank = sv.len() == b && sv.iter().all(|&s| s > 1.0e-9);
            if full_rank && u.ncols() == b {
                for r in 0..b {
                    for c in 0..p {
                        decoder[[g * b + r, c]] = u[[c, r]] as f32;
                    }
                }
            }
        }
        // A degenerate cross-moment (rank-deficient) leaves the block's current
        // (already orthonormal) frame in place; revival handles a truly dead block.
    }
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
/// Returns the number of blocks revived (0 leaves the decoder untouched).
fn revive_dead_blocks(
    x: ArrayView2<'_, f32>,
    codes: &[RowBlockCode],
    decoder: &mut Array2<f32>,
    gamma: f32,
    n_blocks: usize,
    b: usize,
    aux_k: usize,
) -> usize {
    if aux_k == 0 {
        return 0;
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
        return 0;
    }

    // Per-row residual energy under the current model.
    let mut resid = Array2::<f32>::zeros((n, p));
    let mut resid_norm2 = vec![0.0f64; n];
    for i in 0..n {
        let xi = x.row(i);
        let code = &codes[i];
        let selected: Vec<u32> = code
            .blocks
            .iter()
            .zip(code.gates.iter())
            .filter(|&(_, &gate)| gate != 0.0)
            .map(|(&g, _)| g)
            .collect();
        let recon = reconstruct_row(xi, decoder.view(), &selected, gamma, b);
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

    let mut revived = 0usize;
    let mut cursor = 0usize;
    for &g in candidates.iter() {
        // Take the next b distinct high-residual rows for this block's frame.
        if cursor >= n || resid_norm2[row_order[cursor]] <= 1.0e-12 {
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
        for r in 0..b {
            for c in 0..p {
                decoder[[g * b + r, c]] = seed[[r, c]];
            }
        }
        revived += 1;
    }
    revived
}

/// Held-in explained variance `1 − RSS/TSS` of the block reconstruction.
fn explained_variance(
    x: ArrayView2<'_, f32>,
    codes: &[RowBlockCode],
    decoder: ArrayView2<'_, f32>,
    gamma: f32,
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
        let selected: Vec<u32> = code
            .blocks
            .iter()
            .zip(code.gates.iter())
            .filter(|&(_, &gate)| gate != 0.0)
            .map(|(&g, _)| g)
            .collect();
        let recon = reconstruct_row(xi, decoder, &selected, gamma, b);
        for c in 0..p {
            let r = xi[c] as f64 - recon[c] as f64;
            rss += r * r;
            let t = xi[c] as f64 - means[c];
            tss += t * t;
        }
    }
    if tss <= 1.0e-24 {
        if rss <= 1.0e-24 { 1.0 } else { 0.0 }
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
        let selected: Vec<u32> = code
            .blocks
            .iter()
            .zip(code.gates.iter())
            .filter(|&(_, &gate)| gate != 0.0)
            .map(|(&block, _)| block)
            .collect();
        acc += row_loss(xi, decoder, &selected, gamma, b);
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

/// Deterministically seed the block frames from a farthest-point pass over the
/// rows (reusing the atom lane's [`super::update::seed_decoder`] to pick `K`
/// distinct direction rows), then orthonormalise each block's `b` rows so every
/// frame starts as a genuine `St(b, P)` point.
pub(super) fn seed_frames(x: ArrayView2<'_, f32>, n_blocks: usize, b: usize) -> Array2<f32> {
    let k = n_blocks * b;
    let mut decoder = super::update::seed_decoder(x, k);
    for g in 0..n_blocks {
        let mut block = decoder.slice(ndarray::s![g * b..g * b + b, ..]).to_owned();
        orthonormalize_block(&mut block);
        for r in 0..b {
            for c in 0..decoder.ncols() {
                decoder[[g * b + r, c]] = block[[r, c]];
            }
        }
    }
    decoder
}

fn validate(x: ArrayView2<'_, f32>, config: &BlockSparseConfig) -> Result<(), String> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err("fit_block_sparse_dictionary requires a non-empty N×P matrix".to_string());
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err("fit_block_sparse_dictionary input must be finite".to_string());
    }
    if config.n_blocks == 0 {
        return Err("fit_block_sparse_dictionary requires n_blocks >= 1".to_string());
    }
    if config.block_size == 0 {
        return Err("fit_block_sparse_dictionary requires block_size >= 1".to_string());
    }
    if config.block_size > x.ncols() {
        return Err(format!(
            "fit_block_sparse_dictionary block_size b={} cannot exceed output dim P={} \
             (a block's b orthonormal rows must fit in ℝ^P)",
            config.block_size,
            x.ncols()
        ));
    }
    if config.block_topk == 0 {
        return Err("fit_block_sparse_dictionary requires block_topk >= 1".to_string());
    }
    if config.max_epochs == 0 {
        return Err("fit_block_sparse_dictionary requires max_epochs >= 1".to_string());
    }
    if !(config.frame_ridge.is_finite() && config.frame_ridge >= 0.0) {
        return Err("fit_block_sparse_dictionary frame_ridge must be finite and >= 0".to_string());
    }
    if !config.tolerance.is_finite() {
        return Err("fit_block_sparse_dictionary tolerance must be finite".to_string());
    }
    Ok(())
}

/// Fit a block-sparse dictionary to `x` (`N×P`): `G` blocks of `b` orthonormal
/// atoms, block-TopK routing by group ℓ₂ gate, tied signed codes with one shared
/// scalar `γ`, Stiefel-constrained frames refreshed by polar steps, and AuxK
/// dead-block revival. Never forms a dense `N×K` object.
pub fn fit_block_sparse_dictionary(
    x: ArrayView2<'_, f32>,
    config: &BlockSparseConfig,
) -> Result<BlockSparseFit, String> {
    validate(x, config)?;
    let n = x.nrows();
    let g = config.n_blocks;
    let b = config.block_size;
    let k = config.block_topk.min(g).max(1);

    let mut decoder = seed_frames(x, g, b);
    let mut gamma = 1.0f32;

    let mut codes = route_and_code_all(
        x,
        decoder.view(),
        gamma,
        g,
        b,
        k,
        config.minibatch,
        config.block_tile,
    )?;

    let mut prev_ev = f64::NEG_INFINITY;
    let mut converged = false;
    let mut epochs_run = 0usize;

    for epoch in 0..config.max_epochs {
        epochs_run = epoch + 1;

        // (γ) closed-form shared scalar given current frames/routing.
        gamma = refresh_gamma(x, &codes, decoder.view(), b, gamma);

        // (frames) MOD cross-moment + polar reprojection onto St(b, P).
        refresh_frames(x, &codes, &mut decoder, gamma, g, b, config.frame_ridge);

        // (revival) AuxK dead-block reseeding from worst-residual rows.
        let revived = revive_dead_blocks(x, &codes, &mut decoder, gamma, g, b, config.aux_k);

        // (re-encode) fresh routing + codes against the refreshed frames — these
        // define the post-epoch model, feed the next epoch, and score EV.
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

        let ev = explained_variance(x, &codes, decoder.view(), gamma, b);
        let improve = ev - prev_ev;
        // As in the atom lane: do not declare convergence while dead blocks are
        // still being revived (a plateau mid-population would freeze a still-dead
        // tail). Once no block is revived, the ordinary EV-plateau test governs.
        if revived == 0 && improve.abs() <= config.tolerance && epoch > 0 {
            converged = true;
            break;
        }
        prev_ev = ev;
    }

    // One final γ refresh against the last routing so the returned scalar is the
    // exact least-squares fit to the returned frames + codes.
    gamma = refresh_gamma(x, &codes, decoder.view(), b, gamma);
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
    }
    let final_ev = explained_variance(x, &codes, decoder.view(), gamma, b);
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
        converged,
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
