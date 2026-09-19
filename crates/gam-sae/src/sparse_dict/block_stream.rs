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
//! epoch's accumulated per-block projector moments (`M_g`, `P×3b`), the streaming γ
//! numerator/denominator, per-block usage + within-block code second moments (for
//! the utilisation / stable-rank report), the streaming TSS/RSS moments, and the
//! worst-reconstructed-row reservoir feeding AuxK dead-block birth proposals. A shard
//! round-trips only its own rows through Python — never the `K×P` frames or any
//! `N×K` object. The moment storage is `O(K×P)`; temporary projections and sparse
//! block-row incidence lists use `O(B×P + B×k×b + G)` for one configured
//! minibatch of B rows, in addition to the bounded routing score tiles.
//!
//! During an epoch the block frames and the shared
//! scalar γ are FROZEN at their epoch-start values; every shard is routed against
//! them and its cross-moment / γ / moment contributions are summed (all additive),
//! so shard boundaries do not change the accumulated problem. Gamma-free data
//! and overlap moments let the frame step use the newly fitted γ without
//! replaying the corpus. With supports held fixed, a per-block Rayleigh surrogate
//! majorizes the simultaneous tied-code reconstruction loss, differentiating both
//! the frame and its projection code, through the Cauchy–Schwarz bound over the
//! blocks each row admits. Each pass accumulates the surrogate's
//! operator on the frame and on the two search directions the previous step
//! left, and the frame step takes each block's top Ritz vectors on their span:
//! block LOBPCG with its residual one pass behind, with no spectral shift and no
//! step length. Rerouting on the next pass can change supports and is measured
//! separately. One-shot uses sequential block updates; the streaming trajectory
//! is different.
//!
//! Each row carries its support out of the last committed pass. A routed support
//! replaces it only when that lowers the row's tied loss beyond both losses'
//! rounding, so the support step descends the objective the γ and frame steps
//! descend and a row changes support only finitely often (#2502, the atom lane's
//! #2283 rule). Supports are kept by position, so every epoch must stream the same
//! rows in the same order, and a pass that does not is refused. They cost `N·k`
//! u32 slots and one u64 fingerprint per row.
//!
//! EV describes the pass's measured frames with their profiled γ. Proposed frames
//! remain an uncertified checkpoint until a paired pass reroutes both proposal
//! and baseline and profiles a gamma for each. Only a strict full-objective
//! improvement commits, measured over the rows whose codes can differ between the two
//! passes, since every other row is priced to the same bits in both; rejection retracts
//! halfway along the same frame direction and cannot certify stationarity. Finalize
//! requires EV, γ, projector closure and tangent stationarity and returns the exact measured frames,
//! γ and EV; an EV coincidence alone never certifies an unmeasured proposal.

use super::BlockSparseConfig;
use super::block::{
    RowBlockCode, block_birth_evidence_margin, gram_schmidt_rows, relative_scalar_change,
    StoredSpans, route_and_code_all, span_coordinates, stable_rank_symmetric, stored_spans,
};
use super::block_frame::{STORED_FRAME_RESOLUTION, ritz_tied_frame_step, stored_projector_distance};
use super::residual_reservoir::{ResidualReservoir, residual_rounding_energy};
use super::update::DecoderSolveStats;
use gam_linalg::faer_ndarray::with_faer_sequential;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::hash::{DefaultHasher, Hash, Hasher};

/// One minibatch row's gamma-free reconstruction and scalar moments. Keeping
/// only the sum avoids retaining a P-vector for every selected block.
struct RowProjection {
    sum: Vec<f64>,
    rss: f64,
    gamma_num: f64,
    gamma_den: f64,
    /// Live `(block, axis)` terms summed into `sum`.
    live_terms: usize,
    /// `Σ_t |w_t|`, the mass of `sum` over orthonormal frame rows.
    projection_mass: f64,
}

fn row_projection(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    code: &RowBlockCode,
    b: usize,
    gamma: f32,
) -> RowProjection {
    let p = row.len();
    let mut sum = vec![0.0; p];
    let mut projection = vec![0.0; p];
    let mut live_terms = 0usize;
    let mut projection_mass = 0.0f64;
    for (slot, &block) in code.blocks.iter().enumerate() {
        if code.gates[slot] == 0.0 {
            continue;
        }
        projection.fill(0.0);
        let w = &code.projections[slot * b..(slot + 1) * b];
        for (axis, &weight) in w.iter().enumerate() {
            live_terms += 1;
            projection_mass += weight.abs();
            let atom = decoder.row(block as usize * b + axis);
            for (value, &direction) in projection.iter_mut().zip(atom.iter()) {
                *value += weight * direction as f64;
            }
        }
        for (value, contribution) in sum.iter_mut().zip(&projection) {
            *value += contribution;
        }
    }
    let mut rss = 0.0;
    let mut gamma_num = 0.0;
    let mut gamma_den = 0.0;
    for (&x, &projected) in row.iter().zip(&sum) {
        let residual = x as f64 - gamma as f64 * projected;
        rss += residual * residual;
        gamma_num += x as f64 * projected;
        gamma_den += projected * projected;
    }
    RowProjection {
        sum,
        rss,
        gamma_num,
        gamma_den,
        live_terms,
        projection_mass,
    }
}

/// A retained-support slot that holds no block.
const NO_BLOCK: u32 = u32::MAX;

/// A row's identity across passes: a hash of its stored bits. A retained support
/// belongs to the row streamed at the same position only when the hashes match.
fn row_fingerprint(row: ArrayView1<'_, f32>) -> u64 {
    let mut hasher = DefaultHasher::new();
    for value in row {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

/// Append each code's admitted blocks, in slot order, as one `k`-slot support
/// padded with [`NO_BLOCK`].
fn append_supports(supports: &mut Vec<u32>, codes: &[RowBlockCode], k: usize) {
    for code in codes {
        let start = supports.len();
        supports.extend(
            code.blocks
                .iter()
                .zip(&code.gates)
                .filter(|(_, gate)| **gate != 0.0)
                .map(|(block, _)| *block),
        );
        supports.resize(start + k, NO_BLOCK);
    }
}

/// The tied code of a row's retained support `prior` at `decoder` and `gamma`:
/// each retained block's span coordinates `w_g = (U_gU_gᵀ)⁻¹U_g x`, in the retained
/// slot order and padded to `k` slots, with gate `‖P_g x‖`. A retained block the row
/// no longer projects onto takes a zero gate and is not admitted. `inverse_grams` is the
/// `inverse_grams` of `stored_spans(decoder, b)`.
fn kept_block_code(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    inverse_grams: &[f64],
    gamma: f32,
    b: usize,
    k: usize,
    prior: &[u32],
) -> RowBlockCode {
    let gamma64 = gamma as f64;
    let mut blocks = Vec::with_capacity(k);
    let mut gates = Vec::with_capacity(k);
    let mut codes = Vec::with_capacity(k * b);
    let mut projections = Vec::with_capacity(k * b);
    let mut inner = vec![0.0f64; b];
    let mut coordinates = vec![0.0f64; b];
    for &block in prior.iter().take_while(|&&block| block != NO_BLOCK) {
        let base = block as usize * b;
        for (axis, value) in inner.iter_mut().enumerate() {
            let mut projection = 0.0f64;
            for (x, direction) in row.iter().zip(decoder.row(base + axis).iter()) {
                projection += *x as f64 * *direction as f64;
            }
            *value = projection;
        }
        span_coordinates(
            &inverse_grams[base * b..(base + b) * b],
            &inner,
            &mut coordinates,
        );
        let mut energy = 0.0f64;
        for (&product, &coordinate) in inner.iter().zip(&coordinates) {
            energy += product * coordinate;
            projections.push(coordinate);
            codes.push((gamma64 * coordinate) as f32);
        }
        blocks.push(block);
        gates.push(energy.sqrt() as f32);
    }
    while blocks.len() < k {
        blocks.push(0);
        gates.push(0.0);
        for _ in 0..b {
            codes.push(0.0);
            projections.push(0.0);
        }
    }
    RowBlockCode {
        blocks,
        gates,
        codes,
        projections,
    }
}

/// The code a row carries through a pass, with its projection: the freshly routed
/// `fresh` code when its tied loss lies below that of the row's retained support
/// `prior`, re-coded at `decoder`, by more than both losses' rounding; that re-coded
/// retained support otherwise. An empty `prior` (no committed pass yet) adopts
/// `fresh`.
///
/// Greedy admission from the gate shortlist is not the loss minimizer among
/// overlapping blocks, so near-tied blocks can trade a row between them pass after
/// pass while every frame step still descends at fixed supports, and the frames of
/// the small blocks those rows move between never settle (#2502). Keeping a support
/// unless a routed one lowers the row's loss by an amount its arithmetic resolves
/// makes the support step descend the same tied objective as the frame and γ steps,
/// so a row changes support only finitely often: #2283's rule for the atom lane.
///
/// The third value is the routed loss minus the retained loss when the routed support
/// admits different blocks and is declined, and `None` otherwise. `spans` is
/// `stored_spans` of `decoder`.
fn descend_block_support(
    row: ArrayView1<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    spans: &StoredSpans,
    gamma: f32,
    b: usize,
    k: usize,
    fresh: RowBlockCode,
    prior: &[u32],
) -> (RowBlockCode, RowProjection, Option<f64>) {
    let fresh_projection = row_projection(row, decoder, &fresh, b, gamma);
    if prior.first().is_none_or(|&block| block == NO_BLOCK) {
        return (fresh, fresh_projection, None);
    }
    let kept = kept_block_code(row, decoder, &spans.inverse_grams, gamma, b, k, prior);
    let kept_projection = row_projection(row, decoder, &kept, b, gamma);
    if same_admitted_blocks(&fresh, &kept) {
        return (kept, kept_projection, None);
    }
    let row_norm = row
        .iter()
        .map(|&x| x as f64 * x as f64)
        .sum::<f64>()
        .sqrt();
    // Each loss is `‖r‖²` of a residual that rounds within `residual_band` (the
    // reservoir's currency for the same residual), so the loss itself rounds within
    // `2‖r‖·band + band²`.
    let rounding = |code: &RowBlockCode, projection: &RowProjection| {
        let band = residual_band(row_norm, code, projection, spans, b, gamma);
        2.0 * projection.rss.sqrt() * band + band * band
    };
    if fresh_projection.rss + rounding(&fresh, &fresh_projection)
        < kept_projection.rss - rounding(&kept, &kept_projection)
    {
        (fresh, fresh_projection, None)
    } else {
        let excess = fresh_projection.rss - kept_projection.rss;
        (kept, kept_projection, Some(excess))
    }
}

/// The Euclidean band within which a row's residual `x − γ Σ_g U_gᵀw_g`, coded as
/// `code` and projected as `projection`, rounds: the residual's own formation from its
/// coordinates (`residual_rounding_energy`), plus the rounding of the coordinates
/// themselves, which each admitted block's conditioning amplifies
/// (`StoredSpans::coordinate_rounding`). `spans` is `stored_spans` of the pass's decoder.
fn residual_band(
    row_norm: f64,
    code: &RowBlockCode,
    projection: &RowProjection,
    spans: &StoredSpans,
    b: usize,
    gamma: f32,
) -> f64 {
    let scale = (gamma as f64).abs();
    let formation = residual_rounding_energy(
        gam_linalg::roundoff::UNIT_ROUNDOFF,
        3 * projection.live_terms + 2,
        row_norm,
        scale * projection.projection_mass,
    )
    .sqrt();
    formation + scale * spans.coordinate_rounding(code, b, projection.sum.len(), row_norm)
}

/// The routed supports one minibatch declined: rows whose routed support admits
/// different blocks than their retained support without lowering their loss beyond
/// rounding, and the sum of those rows' routed loss minus retained loss at the pass's
/// frames and γ.
#[derive(Default)]
struct DeclinedRoutes {
    rows: usize,
    excess: f64,
}

/// [`descend_block_support`] for every routed row of one minibatch whose first row
/// is the pass's `offset`-th. `retained` holds the last committed pass's `k`-slot
/// supports and is empty before any pass has committed.
fn descend_supports(
    rows: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    spans: &StoredSpans,
    gamma: f32,
    b: usize,
    k: usize,
    routed: Vec<RowBlockCode>,
    retained: &[u32],
    offset: usize,
) -> (Vec<RowBlockCode>, Vec<RowProjection>, DeclinedRoutes) {
    let arbitrated: Vec<(RowBlockCode, RowProjection, Option<f64>)> = routed
        .into_par_iter()
        .enumerate()
        .map(|(row, fresh)| {
            let prior = if retained.is_empty() {
                &[][..]
            } else {
                &retained[(offset + row) * k..(offset + row + 1) * k]
            };
            descend_block_support(
                rows.row(row),
                decoder,
                spans,
                gamma,
                b,
                k,
                fresh,
                prior,
            )
        })
        .collect();
    let mut codes = Vec::with_capacity(arbitrated.len());
    let mut projections = Vec::with_capacity(arbitrated.len());
    let mut declined = DeclinedRoutes::default();
    for (code, projection, excess) in arbitrated {
        if let Some(excess) = excess {
            declined.rows += 1;
            declined.excess += excess;
        }
        codes.push(code);
        projections.push(projection);
    }
    (codes, projections, declined)
}

/// Stable sparse incidence lists let one worker own each block's moments.
/// Rows remain in input order, independent of the worker count or shard cuts.
fn block_row_postings(codes: &[RowBlockCode], blocks: usize) -> Vec<Vec<(usize, usize)>> {
    let mut postings = vec![Vec::new(); blocks];
    for (row, code) in codes.iter().enumerate() {
        for (slot, &block) in code.blocks.iter().enumerate() {
            if code.gates[slot] != 0.0 {
                postings[block as usize].push((row, slot));
            }
        }
    }
    postings
}

/// Whether two codes admit the same set of blocks, in any slot order.
fn same_admitted_blocks(left: &RowBlockCode, right: &RowBlockCode) -> bool {
    let admitted = |code: &RowBlockCode| {
        let mut blocks: Vec<u32> = code
            .blocks
            .iter()
            .zip(&code.gates)
            .filter(|(_, gate)| **gate != 0.0)
            .map(|(block, _)| *block)
            .collect();
        blocks.sort_unstable();
        blocks
    };
    admitted(left) == admitted(right)
}

/// Whether two codes admit the same blocks in the same slots. In a paired frame trial a
/// row whose two codes agree slot for slot on blocks the proposal did not move is coded,
/// decoded and priced by the same operations on the same bits in both passes.
fn same_slot_support(left: &RowBlockCode, right: &RowBlockCode) -> bool {
    let admitted = |code: &RowBlockCode| {
        code.blocks
            .iter()
            .zip(&code.gates)
            .filter(|(_, gate)| **gate != 0.0)
            .map(|(block, _)| *block)
            .collect::<Vec<u32>>()
    };
    admitted(left) == admitted(right)
}

/// One selected row's contribution to its block's tied projector moments over
/// `W = [U, D]`: `coupling += v (Wᵀx)ᵀ + x (Wᵀv)ᵀ − m x (Wᵀx)ᵀ` and
/// `data_cross += x (Wᵀx)ᵀ`, where `v = m P_g x − Σ_h P_h x` is formed from the
/// row's span coordinates `w` (`P_g x = Uᵀw`), its gamma-free reconstruction `sum`,
/// and `m`, the number of blocks the row admits this pass (see `end_epoch`).
///
/// The moments are `H` applied to the stored rows, so they take the inner products
/// `Wᵀx` with those rows, summed here. The span coordinates equal `Ux` only for exactly
/// orthonormal rows, which `f32` storage never holds (see
/// `super::block::stored_span_inverse_gram`).
///
/// `frame` holds the block's `b` directions and `directions` its `d` search rows,
/// both f64 and row-major (`frame[axis * p + c]`); `d` may be zero. `coupling` and
/// `data_cross` are the block's `P×n` moments, row-major (`[c * n + column]`) with
/// `n ≥ b + d`, and only their first `b + d` columns are written. `v` (length `P`)
/// and `coordinates` (length `2(b + d)`) are caller-owned scratch. Every column
/// receives the same floating-point operations in the same order (#2826).
fn accumulate_tied_row_moments(
    frame: &[f64],
    directions: &[f64],
    w: &[f64],
    xi: ArrayView1<'_, f32>,
    sum: &[f64],
    majorizer_weight: f64,
    v: &mut [f64],
    coordinates: &mut [f64],
    coupling: &mut [f64],
    data_cross: &mut [f64],
) {
    let b = w.len();
    let p = sum.len();
    let searched = directions.len() / p;
    let columns = coupling.len() / p;
    coordinates.fill(0.0);
    // `v_coordinates` holds Wᵀv and `x_coordinates` holds Wᵀx, over all `b + d` columns.
    let (v_coordinates, x_coordinates) = coordinates.split_at_mut(b + searched);
    // Pass 1: v, then W v and W x, each summed in `c` order.
    {
        let (frame_v, search_v) = v_coordinates.split_at_mut(b);
        let (frame_x, search_x) = x_coordinates.split_at_mut(b);
        for (c, ((value, &x), &total)) in v.iter_mut().zip(xi.iter()).zip(sum).enumerate() {
            let mut own = 0.0;
            for (axis, &weight) in w.iter().enumerate() {
                own += weight * frame[axis * p + c];
            }
            *value = majorizer_weight * own - total;
            let x = x as f64;
            for (axis, (v_coordinate, x_coordinate)) in
                frame_v.iter_mut().zip(frame_x.iter_mut()).enumerate()
            {
                let direction = frame[axis * p + c];
                *v_coordinate += direction * *value;
                *x_coordinate += direction * x;
            }
            for (row, (v_coordinate, x_coordinate)) in
                search_v.iter_mut().zip(search_x.iter_mut()).enumerate()
            {
                let direction = directions[row * p + c];
                *v_coordinate += direction * *value;
                *x_coordinate += direction * x;
            }
        }
    }
    // Pass 2: each coupling entry takes `v (Wᵀx)`, then `x (Wᵀv)`, then `−m x (Wᵀx)`,
    // as separate additions.
    for (((coupling_row, data_row), &value), &x) in coupling
        .chunks_exact_mut(columns)
        .zip(data_cross.chunks_exact_mut(columns))
        .zip(v.iter())
        .zip(xi.iter())
    {
        let x = x as f64;
        for (column, &projection) in x_coordinates.iter().enumerate() {
            coupling_row[column] += value * projection;
            data_row[column] += x * projection;
            coupling_row[column] += x * v_coordinates[column];
            coupling_row[column] -= majorizer_weight * x * projection;
        }
    }
}

fn profiled_scalar(old_gamma: f32, rss: f64, numerator: f64, denominator: f64) -> (f32, f64) {
    let gamma = if denominator == 0.0 {
        0.0
    } else {
        (numerator / denominator) as f32
    };
    let old = old_gamma as f64;
    let new = gamma as f64;
    let correction = (new - old) * ((new + old) * denominator - 2.0 * numerator);
    (gamma, (rss + correction).max(0.0))
}

/// Grassmann retraction halfway between a rejected proposal and its baseline.
/// Repeated rejection therefore backtracks on the same proposed direction;
/// there is no user-selected damping constant or search box.
///
/// Only the blocks the proposal moved are bisected, and a block keeps the baseline's
/// bits when its midpoint lies within the storage resolution of the baseline, or when
/// storage rounds the midpoint back to the proposal's own bits, so the move cannot be
/// halved in the type the frames are stored in. Re-orthonormalising an unmoved block,
/// or a midpoint the stored rows cannot tell from the baseline, rewrites the same
/// span at `f32` rounding: on the Spark layer-18 fit
/// every one of the 274 frame trials from epoch 2727 to 3000 was a rewrite of all
/// 1024 blocks at projector displacement 6.19e-8, so backtracking never returned to
/// the baseline and every residual stayed infinite (#2502, run 1165180). Here
/// backtracking ends at the baseline itself.
fn bisect_frame_trial(
    baseline: &Array2<f32>,
    proposal: &Array2<f32>,
    b: usize,
) -> Result<Array2<f32>, String> {
    let mut midpoint = baseline.clone();
    for (index, ((mut block, stored), proposed)) in midpoint
        .axis_chunks_iter_mut(Axis(0), b)
        .zip(baseline.axis_chunks_iter(Axis(0), b))
        .zip(proposal.axis_chunks_iter(Axis(0), b))
        .enumerate()
    {
        if stored == proposed {
            continue;
        }
        let mut owned = (&stored + &proposed) * 0.5;
        gram_schmidt_rows(&mut owned);
        let distance = stored_projector_distance(stored, owned.view())
            .map_err(|error| format!("frame trial bisection block {index}: {error}"))?;
        if distance > STORED_FRAME_RESOLUTION && owned != proposed {
            block.assign(&owned);
        }
    }
    Ok(midpoint)
}

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

/// What an adjudicated frame trial measured (#2502). The proposal commits when
/// `decrease` exceeds `resolution`, both over the whole corpus. The `moved_*` fields
/// repeat the comparison over the rows whose admitted blocks, in either pass, include a
/// block the proposal moved. Every other row routes and codes against identical frames
/// at the shared pass γ, so the passes' difference at that γ lives in the moved rows.
#[derive(Clone, Copy, Debug)]
pub struct FrameTrialMeasurement {
    /// Whether the proposal committed: `moved_decrease` exceeded `moved_resolution`.
    pub committed: bool,
    /// Baseline RSS minus proposal RSS over the whole corpus, each at its own profiled γ.
    pub decrease: f64,
    /// `√(rows·p)·ε` times the baseline RSS: the corpus-wide rounding band of `decrease`.
    pub resolution: f64,
    /// The baseline's profiled γ.
    pub baseline_gamma: f32,
    /// The proposal's profiled γ.
    pub proposal_gamma: f32,
    /// Blocks whose stored frames differ between baseline and proposal.
    pub moved_blocks: usize,
    /// The largest relative projector distance among the moved blocks.
    pub moved_displacement: f64,
    /// Rows whose codes can differ between the passes: their admitted blocks, in either pass,
    /// include a moved block, or the two passes admit different blocks or slots. Every other
    /// row is priced to the same bits in both passes.
    pub moved_rows: usize,
    /// Baseline RSS minus proposal RSS over the moved rows, at the pass γ.
    pub moved_decrease: f64,
    /// `√(moved_rows·p)·ε` times the baseline RSS over the moved rows: the bar
    /// `moved_decrease` had to exceed to commit.
    pub moved_resolution: f64,
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
    /// Maximum of relative projector displacement and normalized tangent
    /// gradient at that same gamma. The gradient is read at the measured frames
    /// and does not depend on the step, so a short step cannot manufacture a
    /// stationarity certificate.
    pub frame_residual: f64,
    /// The projector half of [`Self::frame_residual`]: the largest relative
    /// displacement between a block's measured frame and its proposal
    /// (`f64::INFINITY` when no frame step was formed this epoch).
    pub frame_displacement_residual: f64,
    /// The gradient half of [`Self::frame_residual`]: the largest normalized
    /// tangent gradient over blocks (`f64::INFINITY` when no frame step was formed).
    pub frame_gradient_residual: f64,
    /// The block whose own residual is [`Self::frame_residual`], lowest index on
    /// ties; `None` when no frame step was formed this epoch.
    pub frame_binding_block: Option<usize>,
    /// Rows routed to [`Self::frame_binding_block`] this pass (`0` when it is `None`).
    pub frame_binding_block_rows: usize,
    /// Blocks whose own residual exceeds the bar `converged` is tested against,
    /// `tolerance.max(STORED_FRAME_RESOLUTION)`; `None` when no frame step was
    /// formed. It tells one slow block apart from a broadly unconverged dictionary.
    pub frame_blocks_above_tolerance: Option<usize>,
    /// Median over routed blocks of each block's own frame residual, the larger
    /// of its displacement and its normalized gradient (`f64::INFINITY` when no
    /// frame step was formed). Against [`Self::frame_residual`] it tells a broadly
    /// unconverged dictionary from a tail of slow blocks.
    pub frame_residual_median: f64,
    /// Rows whose admitted block set differed between a staged frame proposal
    /// and its baseline in this pass's paired routing; `None` on a pass that
    /// adjudicated no frame trial. The frame step holds supports fixed, so these
    /// rows are the support change it cannot see (#2502).
    pub rerouted_rows: Option<usize>,
    /// What the frame trial this pass adjudicated measured; `None` on a pass that
    /// adjudicated no frame trial.
    pub frame_trial: Option<FrameTrialMeasurement>,
    /// Rows whose committed support this pass differs from the one they carried
    /// out of the last committed pass (every row on the first pass). A row adopts
    /// a routed support only on a loss decrease beyond rounding, so this reaches
    /// zero, and `converged` requires it.
    pub support_changes: usize,
    /// Rows of the live pass whose routed support admitted different blocks than their
    /// retained support and was declined, because it did not lower the row's loss
    /// beyond rounding.
    pub declined_routes: usize,
    /// The EV the live pass would lose by adopting every declined routed support: the
    /// declined rows' routed loss minus retained loss at the pass's frames and γ, over
    /// the pass's total sum of squares.
    pub declined_route_ev_cost: f64,
    /// Mean admitted blocks per row this pass, at most `k`: the per-row
    /// multiplier the frame step's simultaneous-update majorizer charges.
    pub mean_admitted_blocks: f64,
    /// Whether EV, gamma and frame-projector residuals meet the tolerance,
    /// with no accepted or pending block birth.
    pub converged: bool,
    /// Epochs completed so far (this one inclusive).
    pub epoch: usize,
    /// Solve certificate placeholder. The streaming block lane refreshes its frames
    /// by a dense Rayleigh–Ritz step on `3b`-column subspaces (no matrix-free
    /// CG/percolation solve), so this carries the default (zeroed) certificate; the
    /// CG/percolation stats are the atom/dict lane's ([`super::update::DecoderSolveStats`]).
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
    /// The baseline pass's `k`-slot supports, committed if the candidate is rejected.
    baseline_supports: Vec<u32>,
}

/// A frame refresh is not committed until a second, paired streaming pass has
/// priced both the proposal and the frame set which produced it.  Routing is
/// repeated for both dictionaries: fixed-support descent is not evidence of
/// descent after the top-k map changes (#2825).
struct PendingFrameTrial {
    baseline_decoder: Array2<f32>,
    baseline_gamma: f32,
    proposed_decoder: Array2<f32>,
    baseline_rss: f64,
    baseline_gamma_num: f64,
    baseline_gamma_den: f64,
    baseline_rows: usize,
    baseline_usage: Vec<usize>,
    baseline_second: Vec<Array2<f64>>,
    /// The baseline pass's `k`-slot supports, committed if the proposal is rejected.
    baseline_supports: Vec<u32>,
    /// Rows whose admitted block set differs between the proposal and the
    /// baseline on the paired pass.
    rerouted_rows: usize,
    /// The blocks the proposal moved and the rows they reach, from the paired pass.
    moves: TrialMoves,
}

/// The part of a paired frame trial a proposal can change: the blocks whose stored
/// frames differ between baseline and proposal, and the rows whose admitted blocks
/// include one of them, with their baseline and proposal RSS at the pass γ.
#[derive(Default)]
struct TrialMoves {
    /// Per block, whether the proposal moved its frame; set on the first paired minibatch.
    blocks: Option<Vec<bool>>,
    displacement: f64,
    rows: usize,
    baseline_rss: f64,
    proposal_rss: f64,
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
    second: Vec<Array2<f64>>,   // gamma-free projection second moment (b×b)
    normal_second: Vec<Array2<f64>>, // Σ (m − 1) w wᵀ, the majorizer's normal term (b×b)
    coupling: Vec<Array2<f64>>, // (XᵀV + VᵀX)W, V = k X P_g - total projection (P×3b)
    data_cross: Vec<Array2<f64>>, // XᵀX W, W = [U, R, P] (P×3b)
    usage: Vec<usize>,
    alive_count: usize,
    gamma_num: f64,
    gamma_den: f64,
    col_sum: Vec<f64>,
    col_sumsq: Vec<f64>,
    rss: f64,
    row_count: usize,
    admitted_slots: usize, // admitted (row, block) pairs this pass
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
    pending_frame: Option<PendingFrameTrial>,
    // Search rows `[R, P]` per block (`G·2b × P`) left by the last frame step: the
    // Ritz residual and the part of the old frame outside its proposal. The next
    // pass accumulates the surrogate's operator on them alongside U.
    search_directions: Array2<f32>,
    // Each row's `k`-slot support out of the last committed pass (`rows·k`, padded
    // with `NO_BLOCK`) and the fingerprint of the row it belongs to (`rows`); both are
    // empty before the first pass commits. The live pass's arbitrated supports and
    // fingerprints accumulate beside them until `end_epoch` commits one support set.
    retained_supports: Vec<u32>,
    retained_fingerprints: Vec<u64>,
    pass_supports: Vec<u32>,
    pass_fingerprints: Vec<u64>,
    // The live pass's declined routed supports (`DeclinedRoutes`), taken by `end_epoch`.
    pass_declined_routes: usize,
    pass_declined_excess: f64,
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
    /// orthonormal data-row frames (`data_row_frames`); the corpus is streamed
    /// later through [`Self::partial_fit`]. γ starts at 1.
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

        let decoder = super::block::data_row_frames(seed, g, b);

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
            normal_second: (0..g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
            coupling: (0..g).map(|_| Array2::<f64>::zeros((p, 3 * b))).collect(),
            data_cross: (0..g).map(|_| Array2::<f64>::zeros((p, 3 * b))).collect(),
            usage: vec![0; g],
            alive_count: 0,
            gamma_num: 0.0,
            gamma_den: 0.0,
            col_sum: vec![0.0; p],
            col_sumsq: vec![0.0; p],
            rss: 0.0,
            row_count: 0,
            admitted_slots: 0,
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
            pending_frame: None,
            search_directions: Array2::<f32>::zeros((g * 2 * b, p)),
            retained_supports: Vec::new(),
            retained_fingerprints: Vec::new(),
            pass_supports: Vec::new(),
            pass_fingerprints: Vec::new(),
            pass_declined_routes: 0,
            pass_declined_excess: 0.0,
        })
    }

    /// fit_begin with caller-supplied block frames, for experiments that seed their
    /// own dictionary; the supplied decoder is still required to be a `KxP` block
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
            normal_second: (0..g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
            coupling: (0..g).map(|_| Array2::<f64>::zeros((p, 3 * b))).collect(),
            data_cross: (0..g).map(|_| Array2::<f64>::zeros((p, 3 * b))).collect(),
            usage: vec![0; g],
            alive_count: 0,
            gamma_num: 0.0,
            gamma_den: 0.0,
            col_sum: vec![0.0; p],
            col_sumsq: vec![0.0; p],
            rss: 0.0,
            row_count: 0,
            admitted_slots: 0,
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
            pending_frame: None,
            search_directions: Array2::<f32>::zeros((g * 2 * b, p)),
            retained_supports: Vec::new(),
            retained_fingerprints: Vec::new(),
            pass_supports: Vec::new(),
            pass_fingerprints: Vec::new(),
            pass_declined_routes: 0,
            pass_declined_excess: 0.0,
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
        let mut shard_rss = 0.0f64;
        // Only one configured minibatch's projections and sparse incidence
        // lists are live. Parallel workers own disjoint existing block moments;
        // there are no per-worker K×P copies and no shard-sized score/code store.
        for rows in shard.axis_chunks_iter(Axis(0), self.config.minibatch.max(1)) {
            let k = self.k;
            let offset = self.row_count;
            let fingerprints: Vec<u64> = rows
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(row_fingerprint)
                .collect();
            if !self.retained_fingerprints.is_empty() {
                let end = offset + rows.nrows();
                if self.retained_fingerprints.get(offset..end) != Some(&fingerprints[..]) {
                    return Err(format!(
                        "BlockSparseStream.partial_fit: rows {offset}..{end} of this pass are not \
                         the rows the last committed pass streamed at those positions; the stream \
                         keeps each row's support, so every epoch must stream the same rows in the \
                         same order"
                    ));
                }
            }
            // Route, then let each row keep its committed support unless the routed
            // one lowers its loss beyond rounding (`descend_block_support`).
            let route_descending = |decoder: ArrayView2<'_, f32>, scale: f32, spans: &StoredSpans| {
                route_and_code_all(
                    rows,
                    decoder,
                    scale,
                    self.g,
                    b,
                    k,
                    self.config.minibatch,
                    self.config.block_tile,
                )
                .map(|routed| {
                    descend_supports(
                        rows,
                        decoder,
                        spans,
                        scale,
                        b,
                        k,
                        routed,
                        &self.retained_supports,
                        offset,
                    )
                })
            };
            let spans = stored_spans(self.decoder.view(), b)?;
            let (codes, projected, declined) = route_descending(self.decoder.view(), gamma, &spans)?;
            // Route the complete pre-birth model on these same rows before
            // mutating moments. Birth evidence uses a true paired full pass.
            let baseline_codes = self
                .pending_birth
                .as_ref()
                .map(|pending| {
                    stored_spans(pending.baseline_decoder.view(), b)
                        .and_then(|spans| {
                            route_descending(
                                pending.baseline_decoder.view(),
                                pending.baseline_gamma,
                                &spans,
                            )
                        })
                        .map(|arbitrated| (arbitrated.0, arbitrated.1))
                })
                .transpose()?;
            let frame_baseline_codes = self
                .pending_frame
                .as_ref()
                .map(|pending| {
                    stored_spans(pending.baseline_decoder.view(), b)
                        .and_then(|spans| {
                            route_descending(
                                pending.baseline_decoder.view(),
                                pending.baseline_gamma,
                                &spans,
                            )
                        })
                        .map(|arbitrated| (arbitrated.0, arbitrated.1))
                })
                .transpose()?;
            let postings = block_row_postings(&codes, self.g);
            self.admitted_slots += postings.iter().map(Vec::len).sum::<usize>();
            // Each row's admitted-block count: the multiplier of the simultaneous
            // update bound over the blocks that row's reconstruction holds.
            let admitted_counts: Vec<usize> = codes
                .par_iter()
                .map(|code| code.gates.iter().filter(|&&gate| gate != 0.0).count())
                .collect();

            let columns_per_worker = p.div_ceil(rayon::current_num_threads()).max(1);
            self.col_sum
                .par_chunks_mut(columns_per_worker)
                .zip(self.col_sumsq.par_chunks_mut(columns_per_worker))
                .enumerate()
                .for_each(|(chunk, (sums, squares))| {
                    let first = chunk * columns_per_worker;
                    for row in rows.outer_iter() {
                        for (offset, (sum, square)) in
                            sums.iter_mut().zip(squares.iter_mut()).enumerate()
                        {
                            let value = row[first + offset] as f64;
                            *sum += value;
                            *square += value * value;
                        }
                    }
                });

            for (row, projection) in projected.iter().enumerate() {
                shard_rss += projection.rss;
                self.rss += projection.rss;
                self.gamma_num += projection.gamma_num;
                self.gamma_den += projection.gamma_den;
                if aux_on {
                    let residual = rows
                        .row(row)
                        .iter()
                        .zip(&projection.sum)
                        .map(|(&x, &sum)| (x as f64 - gamma as f64 * sum) as f32)
                        .collect();
                    // Each entry costs a product and an addition per term, an addition
                    // per block, and the scaling and subtraction: at most `3t + 2` f64
                    // operations over the `|γ|·Σ_t |w_t|` reconstruction mass; the
                    // coordinates carry their own rounding too (`residual_band`).
                    let row_norm = rows
                        .row(row)
                        .iter()
                        .map(|&x| x as f64 * x as f64)
                        .sum::<f64>()
                        .sqrt();
                    let band = residual_band(row_norm, &codes[row], projection, &spans, b, gamma);
                    let rounding_energy = band * band;
                    self.reservoir.offer(
                        projection.rss,
                        rounding_energy,
                        (self.row_count + row) as u64,
                        residual,
                    );
                }
            }

            self.coupling
                .par_iter_mut()
                .zip(self.data_cross.par_iter_mut())
                .zip(self.second.par_iter_mut())
                .zip(self.normal_second.par_iter_mut())
                .zip(self.usage.par_iter_mut())
                .zip(postings.par_iter())
                .enumerate()
                .for_each(
                    |(block, (((((coupling, data_cross), second), normal_second), usage), entries))| {
                        // An unselected block has nothing to accumulate, and its
                        // usage below adds zero.
                        if entries.is_empty() {
                            return;
                        }
                        // The block's directions are read as f64 once per
                        // minibatch, not cast per feature per selected row.
                        let frame: Vec<f64> = self
                            .decoder
                            .slice(ndarray::s![block * b..(block + 1) * b, ..])
                            .iter()
                            .map(|&value| f64::from(value))
                            .collect();
                        // Before a block's first frame step its search rows are
                        // zero and would add only zeros, so they are skipped.
                        let searched = self
                            .search_directions
                            .slice(ndarray::s![block * 2 * b..(block + 1) * 2 * b, ..]);
                        let directions: Vec<f64> = if searched.iter().all(|&value| value == 0.0) {
                            Vec::new()
                        } else {
                            searched.iter().map(|&value| f64::from(value)).collect()
                        };
                        // Both moments are allocated by `Array2::zeros` and only
                        // mutated in place, so they stay in standard layout.
                        let coupling = coupling
                            .as_slice_mut()
                            .expect("block coupling moments are standard layout");
                        let data_cross = data_cross
                            .as_slice_mut()
                            .expect("block data moments are standard layout");
                        let mut v = vec![0.0; p];
                        let mut coordinates = vec![0.0; 2 * (b + directions.len() / p)];
                        for &(row, slot) in entries {
                            let w = &codes[row].projections[slot * b..(slot + 1) * b];
                            let majorizer_weight = admitted_counts[row] as f64;
                            accumulate_tied_row_moments(
                                &frame,
                                &directions,
                                w,
                                rows.row(row),
                                &projected[row].sum,
                                majorizer_weight,
                                &mut v,
                                &mut coordinates,
                                coupling,
                                data_cross,
                            );
                            for left in 0..b {
                                for right in 0..b {
                                    second[[left, right]] += w[left] * w[right];
                                    normal_second[[left, right]] +=
                                        (majorizer_weight - 1.0) * w[left] * w[right];
                                }
                            }
                        }
                        *usage += entries.len();
                    },
                );

            if let (Some(pending), Some((baseline_codes, baseline))) =
                (self.pending_birth.as_mut(), baseline_codes.as_ref())
            {
                for projection in baseline {
                    pending.baseline_rss += projection.rss;
                }
                append_supports(&mut pending.baseline_supports, baseline_codes, k);
                let baseline_postings = block_row_postings(baseline_codes, self.g);
                pending
                    .baseline_second
                    .par_iter_mut()
                    .zip(pending.baseline_usage.par_iter_mut())
                    .zip(baseline_postings.par_iter())
                    .for_each(|((second, usage), entries)| {
                        for &(row, slot) in entries {
                            let w = &baseline_codes[row].projections[slot * b..(slot + 1) * b];
                            for left in 0..b {
                                for right in 0..b {
                                    second[[left, right]] += (pending.baseline_gamma as f64
                                        * w[left])
                                        * (pending.baseline_gamma as f64 * w[right]);
                                }
                            }
                        }
                        *usage += entries.len();
                    });
                pending.baseline_rows += rows.nrows();
            }
            if let (Some(pending), Some((baseline_codes, baseline))) =
                (self.pending_frame.as_mut(), frame_baseline_codes.as_ref())
            {
                for projection in baseline {
                    pending.baseline_rss += projection.rss;
                    pending.baseline_gamma_num += projection.gamma_num;
                    pending.baseline_gamma_den += projection.gamma_den;
                }
                append_supports(&mut pending.baseline_supports, baseline_codes, k);
                let postings = block_row_postings(baseline_codes, self.g);
                pending
                    .baseline_second
                    .par_iter_mut()
                    .zip(pending.baseline_usage.par_iter_mut())
                    .zip(postings.par_iter())
                    .for_each(|((second, usage), entries)| {
                        for &(row, slot) in entries {
                            let w = &baseline_codes[row].projections[slot * b..(slot + 1) * b];
                            for left in 0..b {
                                for right in 0..b {
                                    second[[left, right]] += w[left] * w[right];
                                }
                            }
                        }
                        *usage += entries.len();
                    });
                pending.baseline_rows += rows.nrows();
                pending.rerouted_rows += codes
                    .iter()
                    .zip(baseline_codes)
                    .filter(|(candidate, baseline)| !same_admitted_blocks(candidate, baseline))
                    .count();
                if pending.moves.blocks.is_none() {
                    let mut moved = vec![false; self.g];
                    for (block, flag) in moved.iter_mut().enumerate() {
                        let baseline_frame = pending
                            .baseline_decoder
                            .slice(ndarray::s![block * b..(block + 1) * b, ..]);
                        let proposal_frame =
                            self.decoder.slice(ndarray::s![block * b..(block + 1) * b, ..]);
                        if baseline_frame != proposal_frame {
                            *flag = true;
                            let distance =
                                stored_projector_distance(baseline_frame, proposal_frame).map_err(
                                    |error| format!("frame trial projector block {block}: {error}"),
                                )?;
                            pending.moves.displacement = pending.moves.displacement.max(distance);
                        }
                    }
                    pending.moves.blocks = Some(moved);
                }
                if let Some(moved) = pending.moves.blocks.as_ref() {
                    let reaches_moved_block = |code: &RowBlockCode| {
                        code.blocks
                            .iter()
                            .zip(&code.gates)
                            .any(|(block, gate)| *gate != 0.0 && moved[*block as usize])
                    };
                    let mut moved_rows = 0usize;
                    let mut moved_baseline_rss = 0.0f64;
                    let mut moved_proposal_rss = 0.0f64;
                    for ((candidate, baseline_code), (proposal_projection, baseline_projection)) in
                        codes
                            .iter()
                            .zip(baseline_codes)
                            .zip(projected.iter().zip(baseline))
                    {
                        if reaches_moved_block(candidate)
                            || reaches_moved_block(baseline_code)
                            || !same_slot_support(candidate, baseline_code)
                        {
                            moved_rows += 1;
                            moved_baseline_rss += baseline_projection.rss;
                            moved_proposal_rss += proposal_projection.rss;
                        }
                    }
                    pending.moves.rows += moved_rows;
                    pending.moves.baseline_rss += moved_baseline_rss;
                    pending.moves.proposal_rss += moved_proposal_rss;
                }
            }
            append_supports(&mut self.pass_supports, &codes, k);
            self.pass_fingerprints.extend(fingerprints);
            self.pass_declined_routes += declined.rows;
            self.pass_declined_excess += declined.excess;
            self.row_count += rows.nrows();
            // Every completed minibatch leaves coherent accumulated state,
            // including when a later minibatch's router returns an error.
            self.alive_count = self.usage.iter().filter(|&&count| count > 0).count();
        }
        // Per-shard heartbeat (#2227): rows in this shard, cumulative rows, the
        // shard reconstruction RSS, live-block count, and the shard wall time. A
        // stalled shard stops advancing this line; under `RUST_LOG=info` a route
        // that never returns is diagnosable immediately rather than after a whole
        // silent epoch.
        log::info!(
            "[SAE block shard] rows={} total_rows={} rss={:.6e} alive_blocks={}/{} \
             shard_s={:.2}",
            shard.nrows(),
            self.row_count,
            shard_rss,
            self.alive_count,
            self.g,
            shard_start.elapsed().as_secs_f64(),
        );
        Ok(BlockShardStats {
            rows: shard.nrows(),
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
        if !self.retained_fingerprints.is_empty()
            && self.row_count != self.retained_fingerprints.len()
        {
            return Err(format!(
                "BlockSparseStream.end_epoch: this pass streamed {} rows but the last committed \
                 pass streamed {}; the stream keeps each row's support, so every epoch must \
                 stream the same rows in the same order",
                self.row_count,
                self.retained_fingerprints.len()
            ));
        }
        // The supports the state carries out of this pass: the live pass's, unless an
        // adjudication below restores a baseline together with its supports.
        let mut committed_supports = std::mem::take(&mut self.pass_supports);

        // EV of the frames routed against this epoch, from the streamed moments.
        let n = self.row_count as f64;
        let mut tss = 0.0f64;
        for c in 0..p {
            tss += self.col_sumsq[c] - self.col_sum[c] * self.col_sum[c] / n;
        }
        let declined_routes = std::mem::take(&mut self.pass_declined_routes);
        let declined_excess = std::mem::take(&mut self.pass_declined_excess);
        let declined_route_ev_cost = if tss > 0.0 {
            declined_excess / tss
        } else {
            0.0
        };
        let mut rejected_frame = false;
        let mut rerouted_rows = None;
        let mut frame_trial = None;
        if let Some(mut trial) = self.pending_frame.take() {
            rerouted_rows = Some(trial.rerouted_rows);
            if trial.baseline_rows != self.row_count {
                return Err(format!(
                    "BlockSparseStream frame trial saw {} candidate rows but {} baseline rows",
                    self.row_count, trial.baseline_rows,
                ));
            }
            let (proposal_gamma, candidate_rss) =
                profiled_scalar(self.gamma, self.rss, self.gamma_num, self.gamma_den);
            let (baseline_gamma, baseline_rss) = profiled_scalar(
                trial.baseline_gamma,
                trial.baseline_rss,
                trial.baseline_gamma_num,
                trial.baseline_gamma_den,
            );
            // The two passes route and code identical rows at the pass γ they share. A row
            // whose two committed codes admit the same blocks in the same slots, none of them
            // moved, is priced to the same bits in both, so the objectives can differ only over
            // the other rows, which the paired pass sums on their own (`TrialMoves`). Only a
            // measured decrease may commit a rerouted proposal (#2634): each of those sums
            // covers `rows · p` cells, so two sums closer than `√cells · ε · RSS` are two
            // roundings of one number, not a direction. A tie inside that band carries no
            // directional information and is handled by backtracking. Over the whole corpus
            // the same band also counts the rounding of every row the passes share. On the
            // Spark layer-18 fit that bar (2.462e-6) refused, from epoch 3792 on, a step whose
            // paired decrease of 2.461e-6 cleared its own bar (7.7e-7) three times over, and
            // the fit cycled at a frame residual of 1.163e-4 against 1e-4 (#2502, lane job
            // 1248199). The profiled corpus decrease and its bar are still reported.
            let cells = (self.row_count * p).max(1) as f64;
            let resolution = cells.sqrt() * f64::EPSILON * baseline_rss.abs();
            let moves = &trial.moves;
            let moved_cells = (moves.rows * p) as f64;
            let moved_decrease = moves.baseline_rss - moves.proposal_rss;
            let moved_resolution = moved_cells.sqrt() * f64::EPSILON * moves.baseline_rss.abs();
            let committed = moved_decrease > moved_resolution;
            frame_trial = Some(FrameTrialMeasurement {
                committed,
                decrease: baseline_rss - candidate_rss,
                resolution,
                baseline_gamma,
                proposal_gamma,
                moved_blocks: moves
                    .blocks
                    .as_ref()
                    .map_or(0, |moved| moved.iter().filter(|&&flag| flag).count()),
                moved_displacement: moves.displacement,
                moved_rows: moves.rows,
                moved_decrease,
                moved_resolution,
            });
            if !committed {
                let midpoint =
                    bisect_frame_trial(&trial.baseline_decoder, &trial.proposed_decoder, b)?;
                self.decoder = trial.baseline_decoder.clone();
                self.gamma = baseline_gamma;
                self.rss = baseline_rss;
                self.usage = std::mem::take(&mut trial.baseline_usage);
                self.second = std::mem::take(&mut trial.baseline_second);
                committed_supports = std::mem::take(&mut trial.baseline_supports);
                // The paired baseline pass accumulated its code second moments
                // without γ, as the live pass does, and the frame step below is what
                // applies γ² to them. A rejected trial skips that step, so the restored
                // moments take the baseline's own profiled γ² here, and every closed
                // epoch stashes γ²-scaled moments for `block_rank_charges`, whose
                // deviance and rank charge read their scale.
                let baseline_gamma_sq = (baseline_gamma as f64) * (baseline_gamma as f64);
                for second in &mut self.second {
                    second.mapv_inplace(|value| value * baseline_gamma_sq);
                }
                self.alive_count = self.usage.iter().filter(|&&count| count > 0).count();
                if midpoint != self.decoder {
                    let baseline_decoder = self.decoder.clone();
                    self.decoder = midpoint.clone();
                    self.pending_frame = Some(PendingFrameTrial {
                        baseline_decoder,
                        baseline_gamma,
                        proposed_decoder: midpoint,
                        baseline_rss: 0.0,
                        baseline_gamma_num: 0.0,
                        baseline_gamma_den: 0.0,
                        baseline_rows: 0,
                        baseline_usage: vec![0; self.g],
                        baseline_second: (0..self.g)
                            .map(|_| Array2::<f64>::zeros((b, b)))
                            .collect(),
                        baseline_supports: Vec::new(),
                        rerouted_rows: 0,
                        moves: TrialMoves::default(),
                    });
                }
                rejected_frame = true;
            }
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
                committed_supports = pending.baseline_supports;
                self.alive_count = self.usage.iter().filter(|&&count| count > 0).count();
                rejected_birth = true;
            }
        }

        let previous_gamma = self.gamma;
        let mut candidate_decoder = self.decoder.clone();
        let mut gamma_residual = f64::INFINITY;
        let mut frame_residual = f64::INFINITY;
        let mut frame_displacement_residual = f64::INFINITY;
        let mut frame_gradient_residual = f64::INFINITY;
        let mut frame_binding_block = None;
        let mut frame_binding_block_rows = 0usize;
        let mut frame_blocks_above_tolerance = None;
        let mut frame_residual_median = f64::INFINITY;
        let frame_bar = self.config.tolerance.max(STORED_FRAME_RESOLUTION);
        if !rejected_birth && !rejected_frame {
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
            // Form the frame proposal at the NEW gamma. Its moments use the
            // same frozen directions and routing as the scalar fit, with no
            // corpus replay and no old-gamma cross term left in the frame step.
            let mut next_directions = Array2::<f32>::zeros((self.g * 2 * b, p));
            // Each task owns one proposal and its moments. Pin nested faer
            // factorizations to sequential while the existing Rayon pool fans
            // out over blocks; otherwise a nested solver barrier can deadlock.
            let outcomes: Vec<Result<f64, String>> = with_faer_sequential(|| {
                self.coupling
                    .par_iter_mut()
                    .zip(self.second.par_iter_mut())
                    .zip(self.normal_second.par_iter_mut())
                    .zip(
                        candidate_decoder
                            .axis_chunks_iter_mut(Axis(0), b)
                            .into_par_iter(),
                    )
                    .zip(
                        next_directions
                            .axis_chunks_iter_mut(Axis(0), 2 * b)
                            .into_par_iter(),
                    )
                    .enumerate()
                    .map(|(gg, ((((moment, second), normal), mut proposal), mut directions))| {
                        second.mapv_inplace(|value| value * gamma * gamma);
                        normal.mapv_inplace(|value| value * gamma * gamma);
                        if self.usage[gg] == 0 {
                            return Ok(0.0);
                        }
                        // For U = Dᵀ, P = UUᵀ and fixed supports, only a row's
                        // admitted blocks S enter its reconstruction, so the bound
                        // ||Σ_{h∈S} ΔP_h x||² <= m Σ_{h∈S} ||ΔP_h x||² with m = |S|
                        // gives the surrogate -tr(U_newᵀ H U_new) with
                        // H = Σ_x (2γ - mγ²) x xᵀ + γ²(x vᵀ + v xᵀ) and
                        // v = m P x - Σ_h P_h x, tight at the stored frame. m is
                        // this pass's admitted count, never k and never a quantity
                        // read off the previous step. The pass accumulated
                        // H·[U, R, P] without γ, and the top Ritz vectors of H on
                        // that span lower the actual tied loss at fixed supports
                        // whenever they move.
                        moment.zip_mut_with(&self.data_cross[gg], |coupling, &data| {
                            *coupling = 2.0 * gamma * data + gamma * gamma * *coupling;
                        });
                        ritz_tied_frame_step(
                            self.decoder.slice(ndarray::s![gg * b..(gg + 1) * b, ..]),
                            self.search_directions
                                .slice(ndarray::s![gg * 2 * b..(gg + 1) * 2 * b, ..]),
                            moment.view(),
                            normal.view(),
                            1.0,
                            frame_bar,
                            proposal.view_mut(),
                            directions.view_mut(),
                        )
                        .map_err(|error| format!("BlockSparseStream frame block {gg}: {error}"))
                    })
                    .collect()
            });
            // Indexed collection preserves the first failing block's identity
            // even if worker completion order changes.
            let gradient = outcomes
                .into_iter()
                .collect::<Result<Vec<f64>, String>>()?;
            self.search_directions = next_directions;
            let displacement = (0..self.g)
                .into_par_iter()
                .map(|block| {
                    stored_projector_distance(
                        self.decoder.slice(ndarray::s![block * b..(block + 1) * b, ..]),
                        candidate_decoder.slice(ndarray::s![block * b..(block + 1) * b, ..]),
                    )
                    .map_err(|error| format!("frame projector block {block}: {error}"))
                })
                .collect::<Result<Vec<f64>, String>>()?;
            frame_displacement_residual = displacement.iter().copied().fold(0.0_f64, f64::max);
            frame_gradient_residual = gradient.iter().copied().fold(0.0_f64, f64::max);
            frame_residual = frame_displacement_residual.max(frame_gradient_residual);
            // Name the block that sets the certificate and count the blocks above the
            // same bar, so one slow block and a broadly unconverged dictionary differ.
            let mut binding: Option<(usize, f64)> = None;
            let mut above = 0usize;
            for (block, (&moved, &tangent)) in displacement.iter().zip(&gradient).enumerate() {
                let residual = moved.max(tangent);
                let binds = match binding {
                    Some((_, best)) => residual > best,
                    None => true,
                };
                if binds {
                    binding = Some((block, residual));
                }
                if residual > frame_bar {
                    above += 1;
                }
            }
            frame_binding_block = binding.map(|(block, _)| block);
            frame_binding_block_rows = binding.map_or(0, |(block, _)| self.usage[block]);
            frame_blocks_above_tolerance = Some(above);
            let mut routed: Vec<f64> = displacement
                .iter()
                .zip(&gradient)
                .zip(&self.usage)
                .filter(|(_, rows)| **rows > 0)
                .map(|((&moved, &tangent), _)| moved.max(tangent))
                .collect();
            routed.sort_by(f64::total_cmp);
            frame_residual_median = routed.get(routed.len() / 2).copied().unwrap_or(0.0);
        }
        let mean_admitted_blocks = self.admitted_slots as f64 / self.row_count as f64;
        let ev = crate::k_selection::explained_variance_within_band(
            self.rss,
            tss,
            crate::k_selection::streamed_tss_rounding_band(
                self.row_count,
                &self.col_sum,
                &self.col_sumsq,
            ),
        );
        // The block lane's Rayleigh–Ritz frames carry no matrix-free CG/percolation
        // certificate (that solver serves the atom/dict lane); report a default.
        let decoder_solve_stats = DecoderSolveStats::default();

        let dead: usize = self.usage.iter().filter(|&&u| u == 0).count();

        // Utilisation + stable-rank report from this epoch's accumulators.
        for gg in 0..self.g {
            self.last_util[gg] = self.usage[gg] as f32 / self.row_count.max(1) as f32;
            self.last_stable[gg] = stable_rank_symmetric(self.second[gg].view());
        }

        // Rows whose committed support differs from the one they carried out of the
        // last committed pass. A kept support is re-coded in its retained slot order and
        // an adopted one differs as a set, so comparing slots counts exactly the rows
        // whose admitted blocks changed.
        let support_changes = if self.retained_supports.is_empty() {
            self.row_count
        } else {
            committed_supports
                .chunks_exact(self.k)
                .zip(self.retained_supports.chunks_exact(self.k))
                .filter(|(committed, retained)| committed != retained)
                .count()
        };
        self.retained_supports = committed_supports;
        self.retained_fingerprints = std::mem::take(&mut self.pass_fingerprints);

        let improve = ev - self.prev_ev;
        let stationary = !rejected_birth
            && !rejected_frame
            && accepted_births == 0
            && support_changes == 0
            && improve.abs() <= self.config.tolerance
            && gamma_residual <= self.config.tolerance
            && frame_residual <= frame_bar
            && self.epochs_run > 0;
        // Certify the frames actually measured in this pass, together with
        // their profiled gamma. A frame proposal needs the next pass before
        // it has either an EV or a gamma certificate of its own.
        if !stationary && !rejected_birth && !rejected_frame {
            let baseline_decoder = self.decoder.clone();
            self.decoder = candidate_decoder.clone();
            if self.decoder != baseline_decoder {
                self.pending_frame = Some(PendingFrameTrial {
                    baseline_decoder,
                    baseline_gamma: self.gamma,
                    proposed_decoder: candidate_decoder,
                    baseline_rss: 0.0,
                    baseline_gamma_num: 0.0,
                    baseline_gamma_den: 0.0,
                    baseline_rows: 0,
                    baseline_usage: vec![0; self.g],
                    baseline_second: (0..self.g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
                    baseline_supports: Vec::new(),
                    rerouted_rows: 0,
                    moves: TrialMoves::default(),
                });
            }
        }
        let birth_pending = self.pending_frame.is_none()
            && !rejected_birth
            && !rejected_frame
            && self.stage_birth_proposal();
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
            frame_displacement_residual,
            frame_gradient_residual,
            frame_binding_block,
            frame_binding_block_rows,
            frame_blocks_above_tolerance,
            frame_residual_median,
            rerouted_rows,
            frame_trial,
            support_changes,
            declined_routes,
            declined_route_ev_cost,
            mean_admitted_blocks,
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
            // The reservoir admits only rows that clear their rounding energy.
            if ranked.len() < b {
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
            baseline_supports: Vec::new(),
        });
        true
    }

    fn reset_epoch(&mut self) {
        for sg in self.second.iter_mut() {
            sg.fill(0.0);
        }
        for normal in self.normal_second.iter_mut() {
            normal.fill(0.0);
        }
        for mg in self.coupling.iter_mut() {
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
        self.admitted_slots = 0;
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
        if !self.converged || self.pending_birth.is_some() || self.pending_frame.is_some() {
            return Err(format!(
                "BlockSparseStream.finalize: streaming fit has not converged after {} epoch(s) \
                 (last EV {:.6e}, EV residual {:.3e}, gamma residual {:.3e}, frame residual {:.3e} \
                 vs tolerance {:.3e}, {} accepted block \
                 birth(s) in the last epoch, birth pending={}, frame trial pending={}); the stream state is a resumable \
                 checkpoint, not a model — run more epochs until end_epoch reports convergence",
                self.epochs_run,
                self.last_ev,
                self.last_ev_residual,
                self.last_gamma_residual,
                self.last_frame_residual,
                self.config.tolerance,
                self.last_accepted_births,
                self.pending_birth.is_some(),
                self.pending_frame.is_some(),
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
            convergence: BlockSparseStreamConvergence {
                corpus_rows: self.last_rows,
                epoch: self.epochs_run,
                ev_residual: self.last_ev_residual,
                gamma_residual: self.last_gamma_residual,
                frame_residual: self.last_frame_residual,
                accepted_births: self.last_accepted_births,
                tolerance: self.config.tolerance,
            },
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

    /// Each row's `k`-slot support out of the last committed pass, by stream position
    /// (`rows·k`, the admitted blocks in slot order, padded with `u32::MAX`). Empty
    /// before the first pass commits.
    pub fn retained_supports(&self) -> &[u32] {
        &self.retained_supports
    }

    /// Epochs closed so far.
    pub fn epochs_run(&self) -> usize {
        self.epochs_run
    }
}

/// The fixed-point certificate a streaming fit was finalized on: the certifying
/// epoch's residuals against the configured tolerance, and the corpus it measured.
#[derive(Clone, Copy, Debug)]
pub struct BlockSparseStreamConvergence {
    /// Rows streamed through the certifying epoch.
    pub corpus_rows: usize,
    /// The certifying epoch (epochs closed so far, inclusive).
    pub epoch: usize,
    /// `|ΔEV|` between the certifying epoch and the epoch before it.
    pub ev_residual: f64,
    /// Relative change of γ in the certifying epoch.
    pub gamma_residual: f64,
    /// The certifying epoch's frame residual.
    pub frame_residual: f64,
    /// Accepted block births in the certifying epoch; zero on a certified fit.
    pub accepted_births: usize,
    /// The configured tolerance the residuals were certified against.
    pub tolerance: f64,
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
    /// Solve certificate placeholder (default/zeroed): the streaming block lane
    /// refreshes its frames by a dense Rayleigh–Ritz step, not the matrix-free
    /// CG/percolation solver that serves the atom/dict lane.
    pub decoder_solve_stats: DecoderSolveStats,
    /// The fixed-point certificate the stream was finalized on.
    pub convergence: BlockSparseStreamConvergence,
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
    if !config.tolerance.is_finite() {
        return Err("BlockSparseStream tolerance must be finite".to_string());
    }
    Ok(())
}

/// Schema tag of a stream checkpoint's fixed little-endian layout. Bump on any layout change,
/// so a checkpoint written by another layout is refused rather than mis-decoded.
const CHECKPOINT_SCHEMA: &[u8; 8] = b"GAMBSS01";

/// A little-endian writer for [`BlockSparseStreamState::checkpoint`].
#[derive(Default)]
struct CheckpointWriter {
    bytes: Vec<u8>,
}

impl CheckpointWriter {
    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    fn usize(&mut self, value: usize) {
        self.bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    fn f32(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn f64(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    fn f32s<'a>(&mut self, values: impl IntoIterator<Item = &'a f32>) {
        for &value in values {
            self.f32(value);
        }
    }
    fn f64s<'a>(&mut self, values: impl IntoIterator<Item = &'a f64>) {
        for &value in values {
            self.f64(value);
        }
    }
    fn u32s(&mut self, values: &[u32]) {
        self.usize(values.len());
        for &value in values {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    fn u64s(&mut self, values: &[u64]) {
        self.usize(values.len());
        for &value in values {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// The reader matching [`CheckpointWriter`]; every read names the checkpoint when it runs out.
struct CheckpointReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> CheckpointReader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self
            .at
            .checked_add(count)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| format!("checkpoint is truncated at byte {}", self.at))?;
        let chunk = &self.bytes[self.at..end];
        self.at = end;
        Ok(chunk)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn usize(&mut self) -> Result<usize, String> {
        let value = u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| "u64 width")?);
        usize::try_from(value).map_err(|_| format!("checkpoint count {value} overflows usize"))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().map_err(|_| "f32 width")?))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().map_err(|_| "f64 width")?))
    }
    fn f32s(&mut self, count: usize) -> Result<Vec<f32>, String> {
        (0..count).map(|_| self.f32()).collect()
    }
    fn f64s(&mut self, count: usize) -> Result<Vec<f64>, String> {
        (0..count).map(|_| self.f64()).collect()
    }
    fn u32s(&mut self) -> Result<Vec<u32>, String> {
        let count = self.usize()?;
        (0..count)
            .map(|_| Ok(u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| "u32 width")?)))
            .collect()
    }
    fn u64s(&mut self) -> Result<Vec<u64>, String> {
        let count = self.usize()?;
        (0..count)
            .map(|_| Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| "u64 width")?)))
            .collect()
    }
    fn matrix_f32(&mut self, rows: usize, columns: usize) -> Result<Array2<f32>, String> {
        Array2::from_shape_vec((rows, columns), self.f32s(rows * columns)?)
            .map_err(|error| error.to_string())
    }
}

impl BlockSparseStreamState {
    /// Write the stream's complete cross-epoch state to `path`, so [`Self::resume`] continues
    /// the fit where it stands: every later epoch of the resumed stream is bit-identical to
    /// the same epoch of this one. A long fit then runs as a chain of bounded jobs (#2502).
    ///
    /// A checkpoint is taken between epochs, after `end_epoch` and before the next
    /// `partial_fit`. Every per-pass accumulator is zero there, and a staged birth or frame
    /// trial holds only its baseline and proposal, so those are all it records; a state with
    /// rows of an unfinished pass streamed is refused. The layout is fixed little-endian under
    /// [`CHECKPOINT_SCHEMA`]. The file is written to a sibling and renamed into place, so an
    /// interrupted write never leaves a truncated checkpoint at `path`.
    pub fn checkpoint(&self, path: &std::path::Path) -> Result<(), String> {
        if self.row_count != 0 {
            return Err(format!(
                "BlockSparseStream.checkpoint: {} rows of an unfinished pass are streamed; a \
                 checkpoint is taken between epochs, after end_epoch",
                self.row_count
            ));
        }
        let mut out = CheckpointWriter::default();
        out.bytes.extend_from_slice(CHECKPOINT_SCHEMA);
        let config = &self.config;
        for value in [
            config.n_blocks,
            config.block_size,
            config.block_topk,
            config.max_epochs,
            config.minibatch,
            config.block_tile,
            config.aux_k,
        ] {
            out.usize(value);
        }
        out.f64(config.frame_ridge);
        out.f64(config.tolerance);
        out.u8(u8::from(config.matryoshka_prefix));
        out.usize(self.p);
        out.f32s(self.decoder.iter());
        out.f32(self.gamma);
        out.f32s(self.search_directions.iter());
        out.f64s(
            [
                self.prev_ev,
                self.last_ev,
                self.last_ev_residual,
                self.last_gamma_residual,
                self.last_frame_residual,
                self.last_rss,
            ]
            .iter(),
        );
        out.usize(self.epochs_run);
        out.usize(self.last_accepted_births);
        out.usize(self.last_rows);
        out.u8(u8::from(self.converged));
        out.f32s(self.last_util.iter());
        out.f32s(self.last_stable.iter());
        for second in &self.last_second {
            out.f64s(second.iter());
        }
        for &usage in &self.last_usage {
            out.usize(usage);
        }
        out.u32s(&self.retained_supports);
        out.u64s(&self.retained_fingerprints);
        match &self.pending_birth {
            None => out.u8(0),
            Some(birth) => {
                out.u8(1);
                out.usize(birth.block);
                out.f32(birth.baseline_gamma);
                out.f32s(birth.baseline_decoder.iter());
            }
        }
        match &self.pending_frame {
            None => out.u8(0),
            Some(trial) => {
                out.u8(1);
                out.f32(trial.baseline_gamma);
                out.f32s(trial.baseline_decoder.iter());
                out.f32s(trial.proposed_decoder.iter());
            }
        }
        let staged = path.with_extension("staged");
        std::fs::write(&staged, &out.bytes)
            .map_err(|error| format!("BlockSparseStream.checkpoint {}: {error}", staged.display()))?;
        std::fs::rename(&staged, path)
            .map_err(|error| format!("BlockSparseStream.checkpoint {}: {error}", path.display()))
    }

    /// Continue a stream from a [`Self::checkpoint`] written with the same `config`. A
    /// checkpoint of another layout, another configuration or a truncated file is refused
    /// by name.
    pub fn resume(path: &std::path::Path, config: &BlockSparseConfig) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("BlockSparseStream.resume {}: {error}", path.display()))?;
        let mut input = CheckpointReader {
            bytes: &bytes,
            at: 0,
        };
        if input.take(CHECKPOINT_SCHEMA.len())? != CHECKPOINT_SCHEMA {
            return Err(format!(
                "BlockSparseStream.resume {}: not a {} checkpoint",
                path.display(),
                String::from_utf8_lossy(CHECKPOINT_SCHEMA)
            ));
        }
        // `max_epochs` bounds the one-shot fit's loop. A stream's epochs are its driver's,
        // so a resumed stream may be driven to another cap and the saved value is skipped.
        for (name, expected) in [
            ("n_blocks", Some(config.n_blocks)),
            ("block_size", Some(config.block_size)),
            ("block_topk", Some(config.block_topk)),
            ("max_epochs", None),
            ("minibatch", Some(config.minibatch)),
            ("block_tile", Some(config.block_tile)),
            ("aux_k", Some(config.aux_k)),
        ] {
            let saved = input.usize()?;
            if let Some(expected) = expected
                && saved != expected
            {
                return Err(format!(
                    "BlockSparseStream.resume: the checkpoint was written with {name}={saved}, \
                     not {expected}"
                ));
            }
        }
        for (name, expected) in [
            ("frame_ridge", config.frame_ridge),
            ("tolerance", config.tolerance),
        ] {
            let saved = input.f64()?;
            if saved.to_bits() != expected.to_bits() {
                return Err(format!(
                    "BlockSparseStream.resume: the checkpoint was written with {name}={saved:e}, \
                     not {expected:e}"
                ));
            }
        }
        let saved_matryoshka = input.u8()? != 0;
        if saved_matryoshka != config.matryoshka_prefix {
            return Err(format!(
                "BlockSparseStream.resume: the checkpoint was written with \
                 matryoshka_prefix={saved_matryoshka}, not {}",
                config.matryoshka_prefix
            ));
        }
        let p = input.usize()?;
        let (g, b) = (config.n_blocks, config.block_size);
        let decoder = input.matrix_f32(g * b, p)?;
        let mut state = Self::new_with_decoder(decoder, config)?;
        state.gamma = input.f32()?;
        state.search_directions = input.matrix_f32(g * 2 * b, p)?;
        let scalars = input.f64s(6)?;
        state.prev_ev = scalars[0];
        state.last_ev = scalars[1];
        state.last_ev_residual = scalars[2];
        state.last_gamma_residual = scalars[3];
        state.last_frame_residual = scalars[4];
        state.last_rss = scalars[5];
        state.epochs_run = input.usize()?;
        state.last_accepted_births = input.usize()?;
        state.last_rows = input.usize()?;
        state.converged = input.u8()? != 0;
        state.last_util = input.f32s(g)?;
        state.last_stable = input.f32s(g)?;
        state.last_second = (0..g)
            .map(|_| {
                Array2::from_shape_vec((b, b), input.f64s(b * b)?).map_err(|error| error.to_string())
            })
            .collect::<Result<_, String>>()?;
        state.last_usage = (0..g).map(|_| input.usize()).collect::<Result<_, String>>()?;
        state.retained_supports = input.u32s()?;
        state.retained_fingerprints = input.u64s()?;
        if state.retained_supports.len() != state.retained_fingerprints.len() * state.k {
            return Err(format!(
                "BlockSparseStream.resume: {} retained support slots for {} rows at k={}",
                state.retained_supports.len(),
                state.retained_fingerprints.len(),
                state.k
            ));
        }
        if input.u8()? == 1 {
            let block = input.usize()?;
            let baseline_gamma = input.f32()?;
            let baseline_decoder = input.matrix_f32(g * b, p)?;
            state.pending_birth = Some(PendingBlockBirth {
                block,
                baseline_decoder,
                baseline_gamma,
                baseline_rss: 0.0,
                baseline_rows: 0,
                baseline_usage: vec![0; g],
                baseline_second: (0..g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
                baseline_supports: Vec::new(),
            });
        }
        if input.u8()? == 1 {
            let baseline_gamma = input.f32()?;
            let baseline_decoder = input.matrix_f32(g * b, p)?;
            let proposed_decoder = input.matrix_f32(g * b, p)?;
            state.pending_frame = Some(PendingFrameTrial {
                baseline_decoder,
                baseline_gamma,
                proposed_decoder,
                baseline_rss: 0.0,
                baseline_gamma_num: 0.0,
                baseline_gamma_den: 0.0,
                baseline_rows: 0,
                baseline_usage: vec![0; g],
                baseline_second: (0..g).map(|_| Array2::<f64>::zeros((b, b))).collect(),
                baseline_supports: Vec::new(),
                rerouted_rows: 0,
                moves: TrialMoves::default(),
            });
        }
        if input.at != bytes.len() {
            return Err(format!(
                "BlockSparseStream.resume {}: {} trailing bytes after the state",
                path.display(),
                bytes.len() - input.at
            ));
        }
        Ok(state)
    }
}

#[cfg(test)]
#[path = "block_stream_tests.rs"]
mod block_stream_tests;
