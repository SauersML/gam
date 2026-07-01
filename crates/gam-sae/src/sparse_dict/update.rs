//! Alternating minibatched trainer: route → sparse codes → decoder refresh →
//! unit-norm projection. No dense `N×K` object is ever formed.
//!
//! The decoder refresh is the **method of optimal directions** (MOD) restricted
//! to the sparse support. With codes fixed, the reconstruction loss
//! `Σ_i ‖x_i − Σ_j c_{ij} d_{a_{ij}}‖²` is quadratic in the decoder `D` and its
//! normal equations are `D (CᵀC + ρI) = CᵀX`, where `C` is the (sparse, never
//! materialised) `N×K` code matrix. We accumulate `A = CᵀC` (`K×K`, but only
//! the few entries touched by co-active atoms are non-zero) and `B = CᵀX`
//! (`K×P`) by streaming minibatches, then solve **exactly**. For a clean
//! `top_s = 1` lane `A` is diagonal and the refresh is a per-atom rescaled mean
//! — exactly MOD / k-SVD's dictionary step. For the general `s > 1` case the
//! coupling `A` is non-diagonal, but it is still SPARSE: only atoms that co-fire
//! in some row are coupled, so after permuting the atoms by the connected
//! components of the co-firing graph `A + ρI` is BLOCK-DIAGONAL. We therefore
//! solve the normal equations EXACTLY (to numerical tolerance) by solving each
//! connected component independently — a small dense SPD Cholesky for ordinary
//! blocks (with a tiny-ridge bump fallback for near-singular blocks, the same
//! robustness as the per-row code solve), or a conjugate-gradient solve iterated
//! to a relative normal-equation residual `‖(A+ρI)x − b‖/‖b‖ ≤ 1e-10` (with a
//! generous iteration cap as a safety backstop) for any rare oversized block.
//! Because distinct components carry no coupling, solving them separately is the
//! exact solution of the full system, and the per-component work keeps the cost
//! off `K²` (each block is only as large as a cluster of mutually co-firing
//! atoms). The `s == 1` / decoupled lane is the singleton-block special case and
//! remains a one-shot exact diagonal solve.

use super::codes::{SparseCode, solve_row_codes};
use super::scoring::TileScorer;
use super::{SparseDictConfig, SparseDictFit};
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use std::collections::HashMap;

/// Route + sparse-code every row of `x`, processing the rows in minibatches of
/// `config.minibatch` so the peak score working set is `minibatch × score_tile`
/// (never `N × K`). Within a minibatch the rows are routed by one batched GEMM
/// per column tile ([`TileScorer::route_minibatch`]) and the per-row active-set
/// code solves run in parallel. The returned `Vec<SparseCode>` is in global row
/// order, identical to a serial row-at-a-time pass up to f32 GEMM rounding.
fn route_and_code_all(
    x: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    scorer: &TileScorer,
    s: usize,
    code_ridge: f32,
    minibatch: usize,
) -> Vec<SparseCode> {
    let n = x.nrows();
    let batch = minibatch.max(1);
    let mut codes: Vec<SparseCode> = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        let end = (start + batch).min(n);
        let block = x.slice(ndarray::s![start..end, ..]);
        let active_lists = route_block(block, decoder, scorer);
        // Per-row code solves are independent; fan them out over the minibatch.
        let mut block_codes: Vec<SparseCode> = block
            .axis_iter(Axis(0))
            .into_par_iter()
            .zip(active_lists.into_par_iter())
            .map(|(row, active)| solve_row_codes(row, decoder, &active, s, code_ridge))
            .collect();
        codes.append(&mut block_codes);
        start = end;
    }
    codes
}

/// Route one minibatch `block` (`B × P`) against `decoder` (`K × P`), returning
/// each row's top-`s` `(atom, score)` shortlist.
///
/// The routing — the `N×K×P` scale-K hot loop the whole lane is built around —
/// is GPU-offloaded when the process admits a device (Linux + CUDA runtime + a
/// `B × K` block above the device break-even), auto-derived at runtime from
/// [`gam_gpu::gpu_mode`] (the charter forbids gating behind a build feature). The
/// device walks `K` in atom-column tiles so peak score memory stays `B × tile`,
/// independent of `K` — the same no-`N×K` discipline as the CPU path. The score
/// block is bit-identical to the CPU per-row `top_s_online` (the kernel forbids
/// FMA contraction), so the GPU routing equals the row-at-a-time oracle exactly.
///
/// The universal fallback (non-Linux, [`gam_gpu::GpuMode::Off`], below
/// break-even, or any device fault under [`gam_gpu::GpuMode::Auto`]) is the
/// parallel CPU GEMM router [`TileScorer::route_minibatch`]. We pass
/// [`gam_gpu::GpuMode::Auto`] — never `Required` — because a real fit must run on
/// any box; the fit is robust to which router ran (it is minibatch-invariant, see
/// [`TileScorer::route_minibatch`]).
fn route_block(
    block: ArrayView2<'_, f32>,
    decoder: ArrayView2<'_, f32>,
    scorer: &TileScorer,
) -> Vec<Vec<(u32, f32)>> {
    #[cfg(target_os = "linux")]
    {
        if gam_gpu::gpu_mode() != gam_gpu::GpuMode::Off {
            // Auto: the router runs on the device when admitted and the block
            // clears the break-even, else it returns the CPU path internally; we
            // only adopt its result when the DEVICE actually produced it, and
            // otherwise use the faster parallel CPU GEMM below.
            if let Ok((routed, super::scoring_gpu::ScoreBlockPath::Device)) =
                super::scoring_gpu::route_minibatch_required(
                    block,
                    decoder,
                    scorer.active,
                    scorer.tile,
                    gam_gpu::GpuMode::Auto,
                )
            {
                return routed;
            }
        }
    }
    scorer.route_minibatch(block, decoder)
}

pub(super) fn run(
    x: ArrayView2<'_, f32>,
    config: &SparseDictConfig,
) -> Result<SparseDictFit, String> {
    validate(x, config)?;
    let n = x.nrows();
    let p = x.ncols();
    let k = config.n_atoms;
    let s = config.active.min(k).max(1);

    let mut decoder = seed_decoder(x, k);
    unit_norm_rows(&mut decoder);

    let scorer = TileScorer::new(s, config.score_tile);
    let mut prev_ev = f64::NEG_INFINITY;
    let mut converged = false;
    let mut epochs_run = 0usize;

    // (a)+(b) route + sparse codes for every row against the seeded, unit-normed
    // decoder, in minibatches: each minibatch is routed by one batched score block
    // per column tile (peak score working set `minibatch × score_tile`, never
    // `N × K`) — on the GPU when the process admits a device and the block clears
    // the break-even, else the parallel CPU GEMM — and its per-row active-set code
    // solves run in parallel. These codes feed the first decoder refresh.
    let mut codes = route_and_code_all(
        x,
        decoder.view(),
        &scorer,
        s,
        config.code_ridge,
        config.minibatch,
    );

    for epoch in 0..config.max_epochs {
        epochs_run = epoch + 1;

        // (c) decoder refresh: solve the sparse decoder normal equations EXACTLY
        // (to tolerance) from the current codes — block-diagonal Cholesky over the
        // co-firing components, see `refresh_decoder`.
        refresh_decoder(x, &codes, &mut decoder, k, p, config);

        // (d) unit-norm projection (identifies code scale) + stable sign.
        unit_norm_rows(&mut decoder);

        // (e) dead-atom revival. Atoms that fired for no row this epoch are re-
        // seeded onto the current worst-reconstructed rows' residual directions.
        // Without this, a large dictionary leaves most atoms at their seed (see
        // the dead counts in the fit report / #1026): effective `K` collapses to a
        // handful of live atoms, EV is non-monotone in `K`, and the lane never
        // climbs toward reconstruction parity. Reviving toward high-residual rows
        // is the standard dead-feature resampling that makes every atom load-
        // bearing, so adding atoms can only help. It runs only while dead atoms
        // remain, so a fully-alive small-`K` dictionary is untouched.
        let revived = revive_dead_atoms(x, &codes, &mut decoder);
        if revived > 0 {
            unit_norm_rows(&mut decoder);
        }

        // (a)+(b) FRESH codes against the just-refreshed, unit-normed decoder.
        // These are the codes that define the post-epoch model, so they (i) feed
        // the NEXT epoch's refresh and (ii) score the convergence EV below. This
        // re-route deliberately replaces the previous STALE-code EV (which scored
        // the new decoder against codes solved before the refresh + normalisation):
        // the convergence decision now uses exactly the codes that define the
        // returned model, so there is no stale-code surrogate gap.
        codes = route_and_code_all(
            x,
            decoder.view(),
            &scorer,
            s,
            config.code_ridge,
            config.minibatch,
        );

        // Convergence-decision EV, computed from the FRESH post-normalisation codes.
        let ev = explained_variance(x, &codes, decoder.view());
        let improve = ev - prev_ev;
        // #1026 — do NOT declare convergence while dead atoms are still being
        // revived. Revival runs at most one atom per row per epoch, so a large
        // dictionary populates its tail over SEVERAL epochs; a one-epoch EV
        // plateau can occur mid-population (a revived atom needs the next route to
        // start firing). Stopping on that plateau froze the dictionary with a
        // still-dead tail — effective `K` below the requested `K`, the exact
        // under-population this issue tracks. Requiring `revived == 0` forces every
        // atom to become load-bearing before the plateau test can fire, so the
        // returned dictionary uses its full budget and its EV climbs with `K`
        // toward reconstruction parity. Once no atom is dead, revival returns 0 and
        // the ordinary EV-plateau test governs stopping exactly as before.
        if revived == 0 && improve.abs() <= config.tolerance && epoch > 0 {
            converged = true;
            break;
        }
        prev_ev = ev;
    }

    // `codes` already hold the re-route against the final, unit-normed decoder
    // (the last epoch's step-(a)+(b)), so the stored codes and the returned EV
    // match the returned dictionary exactly — no extra reroute needed, and the
    // returned EV equals the convergence-decision EV from the final epoch.
    let final_ev = explained_variance(x, &codes, decoder.view());

    let (indices, code_mat) = pack_codes(&codes, n, s);
    Ok(SparseDictFit {
        decoder,
        indices,
        codes: code_mat,
        explained_variance: final_ev,
        epochs: epochs_run,
        converged,
        active: s,
    })
}

fn validate(x: ArrayView2<'_, f32>, config: &SparseDictConfig) -> Result<(), String> {
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err("fit_sparse_dictionary requires a non-empty N×P matrix".to_string());
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err("fit_sparse_dictionary input must be finite".to_string());
    }
    if config.n_atoms == 0 {
        return Err("fit_sparse_dictionary requires K >= 1".to_string());
    }
    if config.active == 0 {
        return Err("fit_sparse_dictionary requires active (top_s) >= 1".to_string());
    }
    if config.max_epochs == 0 {
        return Err("fit_sparse_dictionary requires max_epochs >= 1".to_string());
    }
    if !(config.code_ridge.is_finite() && config.code_ridge >= 0.0) {
        return Err("fit_sparse_dictionary code_ridge must be finite and non-negative".to_string());
    }
    if !(config.decoder_ridge.is_finite() && config.decoder_ridge >= 0.0) {
        return Err(
            "fit_sparse_dictionary decoder_ridge must be finite and non-negative".to_string(),
        );
    }
    if !config.tolerance.is_finite() {
        return Err("fit_sparse_dictionary tolerance must be finite".to_string());
    }
    Ok(())
}

/// Seed atoms with a deterministic k-means++-style farthest-point pass on the
/// rows, so the initial dictionary already spans the data's principal
/// directions (no RNG -> reproducible). For `K > N` the extra atoms wrap.
fn seed_decoder(x: ArrayView2<'_, f32>, k: usize) -> Array2<f32> {
    let n = x.nrows();
    let p = x.ncols();
    let mut decoder = Array2::<f32>::zeros((k, p));

    // First atom: the largest-norm row.
    let mut first = 0usize;
    let mut best = f32::NEG_INFINITY;
    for i in 0..n {
        let r = x.row(i);
        let nrm: f32 = r.iter().map(|v| v * v).sum();
        if nrm > best {
            best = nrm;
            first = i;
        }
    }
    decoder.row_mut(0).assign(&x.row(first));

    let mut min_dist2 = vec![f32::INFINITY; n];
    for atom in 1..k {
        let prev = decoder.row(atom - 1);
        for i in 0..n {
            let mut d2 = 0.0f32;
            let xi = x.row(i);
            for c in 0..p {
                let d = xi[c] - prev[c];
                d2 += d * d;
            }
            if d2 < min_dist2[i] {
                min_dist2[i] = d2;
            }
        }
        let chosen = if atom < n {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for i in 0..n {
                if min_dist2[i] > bv {
                    bv = min_dist2[i];
                    bi = i;
                }
            }
            bi
        } else {
            atom % n
        };
        decoder.row_mut(atom).assign(&x.row(chosen));
    }
    decoder
}

/// The assembled sparse decoder normal equations `(A + ρI) D = B`, with
/// `A = CᵀC` (`K×K`, symmetric PSD) and `B = CᵀX` (`K×P`), where the code matrix
/// `C` is never materialised. Only atom pairs that co-fire in some row appear in
/// `A`, so the coupling is sparse: `diag` holds `A_kk`, `off` holds the strictly
/// upper-triangular couplings `A_{kl}` (`k < l`), and `b` holds `B`.
struct DecoderNormalEq {
    /// `A_kk = Σ_i c_{ik}²`, length `K`.
    diag: Vec<f64>,
    /// `B = CᵀX`, `K×P`.
    b: Array2<f64>,
    /// Off-diagonal couplings `A_{kl}` keyed by `(k, l)` with `k < l`.
    off: HashMap<(u32, u32), f64>,
}

/// Assemble the sparse decoder normal equations `(A + ρI) D = B` from the fixed
/// codes/supports (`ρ` is applied at solve time, so this returns the bare
/// `A`/`B`). Streams the rows once, accumulating only the entries the codes
/// touch — never an `N×K` or dense `K×K` object.
fn assemble_normal_eq(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    k: usize,
    p: usize,
) -> DecoderNormalEq {
    let mut diag = vec![0.0f64; k];
    let mut b = Array2::<f64>::zeros((k, p));
    let mut off: HashMap<(u32, u32), f64> = HashMap::new();

    for (row_idx, code) in codes.iter().enumerate() {
        let xi = x.row(row_idx);
        // Contiguous row slice of the input when possible (standard layout); `None`
        // only for a non-contiguous view, which takes the strided fallback below.
        let xi_slice = xi.as_slice();
        for a in 0..code.indices.len() {
            let ca = code.codes[a] as f64;
            if ca == 0.0 {
                continue;
            }
            let ka = code.indices[a];
            diag[ka as usize] += ca * ca;
            let brow = ka as usize;
            // B_{ka,:} += ca · x_row. Accumulate over contiguous row slices so LLVM
            // autovectorizes the widening f32→f64 FMA — verified in the emitted code
            // as unrolled NEON `fcvtl` + `fmul.2d` + `fadd.2d` with the bounds check
            // hoisted out of the loop (the ndarray 2D-index form defeats that because
            // the stride is opaque). Bit-identical to the strided form (same values,
            // same per-`(brow,c)` accumulation order), so determinism is preserved.
            // `b` is freshly allocated row-major, so its row slice is always `Some`;
            // the strided arm only runs when the *input* view is non-contiguous.
            let mut brow_view = b.row_mut(brow);
            match (brow_view.as_slice_mut(), xi_slice) {
                (Some(bs), Some(xs)) => {
                    for (bref, &xv) in bs.iter_mut().zip(xs.iter()) {
                        *bref += ca * xv as f64;
                    }
                }
                _ => {
                    for c in 0..p {
                        brow_view[c] += ca * xi[c] as f64;
                    }
                }
            }
            for bsel in (a + 1)..code.indices.len() {
                let cb = code.codes[bsel] as f64;
                if cb == 0.0 {
                    continue;
                }
                let kb = code.indices[bsel];
                if ka == kb {
                    // Same atom appearing twice (padding) — fold into diagonal.
                    diag[ka as usize] += 2.0 * ca * cb;
                    continue;
                }
                let key = if ka < kb { (ka, kb) } else { (kb, ka) };
                *off.entry(key).or_insert(0.0) += ca * cb;
            }
        }
    }

    DecoderNormalEq { diag, b, off }
}

/// An atom is "dead" this epoch when its regularised self-energy `A_kk + ρ` is
/// at or below this floor: it never fired (and, since couplings require two
/// non-zero codes, it is then necessarily isolated). Such atoms keep their
/// seeded direction so a later epoch can still route rows to them.
const DEAD_DENOM: f64 = 1.0e-12;

/// Connected-component blocks above this size are solved by conjugate gradient
/// rather than a dense Cholesky, to keep the per-block cost off `O(m³)`/`O(m²)`
/// memory. Mutually co-firing atom clusters are tiny in practice, so the dense
/// path is the norm and this is only a safety valve.
const MAX_DIRECT_BLOCK: usize = 512;

/// Relative normal-equation residual `‖(A+ρI)x − b‖/‖b‖` the CG block solver
/// drives below before stopping — well under the 1e-8 exactness contract.
const CG_REL_TOL: f64 = 1.0e-10;

/// Refresh the decoder by solving the sparse normal equations `(A + ρI) D = B`
/// EXACTLY (to numerical tolerance), not by a fixed number of approximate sweeps.
///
/// `A + ρI` is symmetric PD and, crucially, BLOCK-DIAGONAL once the atoms are
/// grouped by the connected components of the co-firing graph (atoms in distinct
/// components are uncoupled). We therefore solve each component independently:
/// the solution of every block is, jointly, the exact solution of the whole
/// system. Singletons (`s == 1`, or any atom that never co-fires) are a one-shot
/// diagonal solve `D_k = B_k / (A_kk + ρ)`; small components are a dense SPD
/// Cholesky with a tiny-ridge bump fallback for near-singular blocks; rare large
/// components fall back to CG iterated to [`CG_REL_TOL`]. Dead atoms keep their
/// seeded direction. Per-component work keeps the cost off `K²`.
fn refresh_decoder(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    decoder: &mut Array2<f32>,
    k: usize,
    p: usize,
    config: &SparseDictConfig,
) {
    let ridge = config.decoder_ridge as f64;
    let eq = assemble_normal_eq(x, codes, k, p);
    solve_decoder(decoder, &eq, ridge);
}

/// Re-seed atoms that fired for no row this epoch (dead atoms) onto the current
/// worst-reconstructed rows' residual directions — the "dead-feature resampling"
/// that lets a large dictionary actually use all `K` atoms (#1026).
///
/// Pointing a fresh atom at the largest reconstruction error is the greedy step
/// that reduces RSS the most; distinct dead atoms take distinct high-residual
/// rows so revived atoms do not duplicate each other. The residual is computed
/// under the current (just-refreshed, unit-normed) decoder and the `codes` that
/// produced this epoch's routing, so it reflects the live model's error. Only the
/// residual *direction* is installed (raw, un-normed); the caller re-runs the
/// unit-norm + sign projection. At most one atom is revived per distinct row per
/// epoch — with more dead atoms than rows the remainder revive on later epochs as
/// the residual field changes, which is the standard bounded-resample cadence.
///
/// Returns the number of atoms revived (0 leaves the decoder untouched, so a
/// fully-alive small-`K` dictionary pays only the usage scan).
fn revive_dead_atoms(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    decoder: &mut Array2<f32>,
) -> usize {
    let n = x.nrows();
    let p = x.ncols();
    let k = decoder.nrows();

    // Which atoms fired (non-zero code) for at least one row this epoch.
    let mut alive = vec![false; k];
    for code in codes.iter() {
        for (j, &idx) in code.indices.iter().enumerate() {
            if code.codes[j] != 0.0 {
                alive[idx as usize] = true;
            }
        }
    }
    let dead: Vec<usize> = (0..k).filter(|&a| !alive[a]).collect();
    if dead.is_empty() {
        return 0;
    }

    // Per-row residual under the current model, and its squared norm for ranking.
    let mut resid = Array2::<f32>::zeros((n, p));
    let mut resid_norm2 = vec![0.0f64; n];
    for i in 0..n {
        let xi = x.row(i);
        let mut ri = resid.row_mut(i);
        for c in 0..p {
            ri[c] = xi[c];
        }
        let code = &codes[i];
        for j in 0..code.indices.len() {
            let cj = code.codes[j];
            if cj == 0.0 {
                continue;
            }
            let drow = decoder.row(code.indices[j] as usize);
            for c in 0..p {
                ri[c] -= cj * drow[c];
            }
        }
        let mut acc = 0.0f64;
        for c in 0..p {
            acc += ri[c] as f64 * ri[c] as f64;
        }
        resid_norm2[i] = acc;
    }

    // Rows ranked by descending residual energy (ties by ascending index →
    // deterministic). Only rows with real residual can seed a useful atom.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        resid_norm2[b]
            .partial_cmp(&resid_norm2[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });

    let mut revived = 0usize;
    for (t, &atom) in dead.iter().enumerate() {
        if t >= n {
            break; // one atom per distinct row this epoch
        }
        let row = order[t];
        if resid_norm2[row] <= (DEAD_DENOM as f64) {
            break; // remaining rows are already reconstructed — nothing to seed
        }
        let src = resid.row(row);
        let mut dst = decoder.row_mut(atom);
        for c in 0..p {
            dst[c] = src[c];
        }
        revived += 1;
    }
    revived
}

/// Solve `(A + ρI) D = B` exactly, writing the solved rows into `decoder`.
///
/// Atoms are walked in ascending index order and grouped into connected
/// components via BFS over the symmetric coupling adjacency; each component is
/// sorted (canonical order) before solving so the result is bit-reproducible
/// regardless of `HashMap` iteration order. Dead atoms ([`DEAD_DENOM`]) and
/// atoms with no co-firing partner keep / take the trivial solve.
fn solve_decoder(decoder: &mut Array2<f32>, eq: &DecoderNormalEq, ridge: f64) {
    let k = eq.diag.len();
    let p = eq.b.ncols();

    // Symmetric coupling adjacency, sorted per atom for deterministic assembly.
    let mut neigh: Vec<Vec<(u32, f64)>> = vec![Vec::new(); k];
    for (&(a, b), &val) in eq.off.iter() {
        neigh[a as usize].push((b, val));
        neigh[b as usize].push((a, val));
    }
    for list in neigh.iter_mut() {
        list.sort_by_key(|&(nb, _)| nb);
    }

    let mut visited = vec![false; k];
    for start in 0..k {
        if visited[start] {
            continue;
        }
        if neigh[start].is_empty() {
            // Isolated atom: diagonal (singleton) solve, exact in one shot.
            visited[start] = true;
            let denom = eq.diag[start] + ridge;
            if denom <= DEAD_DENOM {
                // Dead atom: keep its seeded direction (no permanent collapse).
                continue;
            }
            for c in 0..p {
                decoder[[start, c]] = (eq.b[[start, c]] / denom) as f32;
            }
            continue;
        }
        // Gather the whole connected component by BFS, then canonicalise order.
        let mut comp = vec![start];
        visited[start] = true;
        let mut head = 0usize;
        while head < comp.len() {
            let node = comp[head];
            head += 1;
            for &(nb, _) in &neigh[node] {
                let nb = nb as usize;
                if !visited[nb] {
                    visited[nb] = true;
                    comp.push(nb);
                }
            }
        }
        comp.sort_unstable();
        solve_component(decoder, eq, ridge, &comp, &neigh, p);
    }
}

/// Solve one connected component's block exactly: dense SPD Cholesky when the
/// block is small ([`MAX_DIRECT_BLOCK`]), else CG. `comp` is the component's
/// atom indices in ascending order; `neigh` is the global sorted adjacency.
fn solve_component(
    decoder: &mut Array2<f32>,
    eq: &DecoderNormalEq,
    ridge: f64,
    comp: &[usize],
    neigh: &[Vec<(u32, f64)>],
    p: usize,
) {
    let m = comp.len();
    // Local atom -> block-row index map (comp is sorted, so this is canonical).
    let mut local: HashMap<usize, usize> = HashMap::with_capacity(m);
    for (i, &a) in comp.iter().enumerate() {
        local.insert(a, i);
    }

    if m <= MAX_DIRECT_BLOCK {
        // Assemble the dense block (A_sub + ρI) and the m×P right-hand side, then
        // solve all P columns from one Cholesky factor.
        let mut mat = Array2::<f64>::zeros((m, m));
        let mut rhs = Array2::<f64>::zeros((m, p));
        for (i, &a) in comp.iter().enumerate() {
            mat[[i, i]] = eq.diag[a] + ridge;
            for &(nb, val) in &neigh[a] {
                if let Some(&j) = local.get(&(nb as usize)) {
                    mat[[i, j]] = val;
                }
            }
            for c in 0..p {
                rhs[[i, c]] = eq.b[[a, c]];
            }
        }
        let sol = cholesky_solve_block(&mat, &rhs);
        for (i, &a) in comp.iter().enumerate() {
            for c in 0..p {
                decoder[[a, c]] = sol[[i, c]] as f32;
            }
        }
        return;
    }

    // Oversized block: solve each column by CG to the residual tolerance. The
    // operator is the component-restricted symmetric mat-vec.
    let matvec = |xloc: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; m];
        for (i, &a) in comp.iter().enumerate() {
            let mut acc = (eq.diag[a] + ridge) * xloc[i];
            for &(nb, val) in &neigh[a] {
                if let Some(&j) = local.get(&(nb as usize)) {
                    acc += val * xloc[j];
                }
            }
            y[i] = acc;
        }
        y
    };
    // Generous safety cap: CG converges in <= m steps in exact arithmetic.
    let cap = m.saturating_mul(20).saturating_add(100);
    for c in 0..p {
        let mut bvec = vec![0.0f64; m];
        let mut bnorm2 = 0.0f64;
        for (i, &a) in comp.iter().enumerate() {
            bvec[i] = eq.b[[a, c]];
            bnorm2 += bvec[i] * bvec[i];
        }
        let bnorm = bnorm2.sqrt();
        let mut xvec = vec![0.0f64; m];
        if bnorm <= DEAD_DENOM {
            for &a in comp {
                decoder[[a, c]] = 0.0;
            }
            continue;
        }
        // CG from a zero start: r0 = b - A*0 = b.
        let mut r = bvec;
        let mut pdir = r.clone();
        let mut rs_old: f64 = r.iter().map(|v| v * v).sum();
        for _ in 0..cap {
            let ap = matvec(&pdir);
            let mut pap = 0.0f64;
            for i in 0..m {
                pap += pdir[i] * ap[i];
            }
            if pap <= 0.0 {
                break; // not happen for SPD; guard against round-off breakdown.
            }
            let alpha = rs_old / pap;
            for i in 0..m {
                xvec[i] += alpha * pdir[i];
                r[i] -= alpha * ap[i];
            }
            let rnorm: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rnorm / bnorm <= CG_REL_TOL {
                break;
            }
            let rs_new: f64 = r.iter().map(|v| v * v).sum();
            let beta = rs_new / rs_old;
            for i in 0..m {
                pdir[i] = r[i] + beta * pdir[i];
            }
            rs_old = rs_new;
        }
        for (i, &a) in comp.iter().enumerate() {
            decoder[[a, c]] = xvec[i] as f32;
        }
    }
}

/// Dense SPD solve `mat · X = rhs` (multiple RHS columns) via Cholesky, with the
/// same tiny-ridge bump fallback as the per-row code solve for near-singular /
/// exactly-collinear blocks, and a diagonal last resort.
fn cholesky_solve_block(mat: &Array2<f64>, rhs: &Array2<f64>) -> Array2<f64> {
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerCholesky;

    let m = mat.nrows();
    let mut a = mat.clone();
    let mut bump = 0.0f64;
    for _attempt in 0..6 {
        if let Ok(factor) = a.cholesky(Side::Lower) {
            return factor.solve_mat(rhs);
        }
        bump = if bump == 0.0 { 1.0e-8 } else { bump * 16.0 };
        a = mat.clone();
        for i in 0..m {
            a[[i, i]] += bump;
        }
    }
    // Degenerate beyond recovery: per-row diagonal solve (independent atoms).
    let p = rhs.ncols();
    let mut out = Array2::<f64>::zeros((m, p));
    for i in 0..m {
        let d = mat[[i, i]].max(DEAD_DENOM);
        for c in 0..p {
            out[[i, c]] = rhs[[i, c]] / d;
        }
    }
    out
}

fn unit_norm_rows(decoder: &mut Array2<f32>) {
    for mut row in decoder.outer_iter_mut() {
        let nrm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if nrm > 1.0e-12 {
            row.mapv_inplace(|v| v / nrm);
            // Orient by first significant component for a stable sign.
            let mut sign = 1.0f32;
            for &v in row.iter() {
                if v.abs() > 1.0e-9 {
                    sign = v.signum();
                    break;
                }
            }
            if sign < 0.0 {
                row.mapv_inplace(|v| -v);
            }
        }
    }
}

fn explained_variance(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    decoder: ArrayView2<'_, f32>,
) -> f64 {
    let n = x.nrows();
    let p = x.ncols();
    // Column means for TSS.
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
    let mut recon = vec![0.0f64; p];
    for i in 0..n {
        for c in 0..p {
            recon[c] = 0.0;
        }
        let code = &codes[i];
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
        let xi = x.row(i);
        for c in 0..p {
            let r = xi[c] as f64 - recon[c];
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

fn pack_codes(codes: &[SparseCode], n: usize, s: usize) -> (Array2<u32>, Array2<f32>) {
    let mut indices = Array2::<u32>::zeros((n, s));
    let mut code_mat = Array2::<f32>::zeros((n, s));
    for (i, code) in codes.iter().enumerate() {
        for j in 0..s {
            indices[[i, j]] = code.indices[j];
            code_mat[[i, j]] = code.codes[j];
        }
    }
    (indices, code_mat)
}

#[cfg(test)]
mod exact_solve_tests {
    use super::{
        DecoderNormalEq, assemble_normal_eq, explained_variance, route_and_code_all, solve_decoder,
    };
    use crate::sparse_dict::codes::SparseCode;
    use crate::sparse_dict::scoring::TileScorer;
    use crate::sparse_dict::{SparseDictConfig, fit_sparse_dictionary};
    use ndarray::Array2;

    impl DecoderNormalEq {
        /// Symmetric sparse mat-vec `y = (A + ρI) x` for one decoder column `x`
        /// (length `K`). Whole-system form used by the exactness tests to measure
        /// the normal-equation residual (the block solver uses a
        /// component-restricted variant inline). Touches only the non-zero
        /// couplings, so it is `O(K + nnz)` and never forms a dense `K×K` matrix.
        fn matvec_col(&self, ridge: f64, x: &[f64]) -> Vec<f64> {
            let k = self.diag.len();
            let mut y = vec![0.0f64; k];
            for i in 0..k {
                y[i] = (self.diag[i] + ridge) * x[i];
            }
            for (&(a, b), &val) in self.off.iter() {
                y[a as usize] += val * x[b as usize];
                y[b as usize] += val * x[a as usize];
            }
            y
        }
    }

    /// A small synthetic decoder-update problem with OVERLAPPING supports (`s = 3`):
    /// five codes whose atom sets slide around the 5-atom dictionary so every atom
    /// fires and many atom pairs co-fire — i.e. the coupled `s > 1` regime, not the
    /// decoupled diagonal one. Returns `(x, codes, k, p)`.
    fn overlapping_problem() -> (Array2<f32>, Vec<SparseCode>, usize, usize) {
        let k = 5usize;
        let p = 4usize;
        // Overlapping 3-atom supports (a sliding window) with generic codes.
        let supports: [[u32; 3]; 5] = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 0],
            [4, 0, 1],
        ];
        let codevals: [[f32; 3]; 5] = [
            [1.0, 0.5, -0.3],
            [0.7, -0.2, 0.4],
            [-0.6, 0.9, 0.1],
            [0.3, -0.5, 0.8],
            [0.2, 0.6, -0.4],
        ];
        let codes: Vec<SparseCode> = supports
            .iter()
            .zip(codevals.iter())
            .map(|(idx, cv)| SparseCode {
                indices: idx.to_vec(),
                codes: cv.to_vec(),
            })
            .collect();
        let n = codes.len();
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            for c in 0..p {
                x[[i, c]] = (((i * 7 + c * 3 + 1) % 13) as f32 - 6.0) / 4.0;
            }
        }
        (x, codes, k, p)
    }

    /// Relative normal-equation residual `‖(A+ρI)D − B‖_F / ‖B‖_F`, summed over all
    /// decoder columns, using the same sparse operator the solver uses.
    fn normal_eq_residual(eq: &DecoderNormalEq, decoder: &Array2<f32>, ridge: f64) -> f64 {
        let k = eq.diag.len();
        let p = eq.b.ncols();
        let mut rss = 0.0f64;
        let mut bss = 0.0f64;
        for c in 0..p {
            let dcol: Vec<f64> = (0..k).map(|i| decoder[[i, c]] as f64).collect();
            let y = eq.matvec_col(ridge, &dcol);
            for i in 0..k {
                let r = y[i] - eq.b[[i, c]];
                rss += r * r;
                bss += eq.b[[i, c]] * eq.b[[i, c]];
            }
        }
        if bss <= 0.0 { 0.0 } else { (rss / bss).sqrt() }
    }

    #[test]
    fn exact_solver_drives_normal_eq_residual_below_tolerance() {
        // The decoder update must solve the coupled normal equations EXACTLY (to
        // tolerance) for s > 1 / overlapping supports — not approximate them with a
        // fixed number of sweeps.
        let (x, codes, k, p) = overlapping_problem();
        let ridge = 1.0e-6f64;
        let eq = assemble_normal_eq(x.view(), &codes, k, p);
        // Guard: the supports really do couple atoms (we are exercising the coupled
        // path, not a disguised diagonal solve).
        assert!(
            !eq.off.is_empty(),
            "test problem must have off-diagonal coupling (overlapping supports)"
        );

        let mut decoder = Array2::<f32>::zeros((k, p));
        solve_decoder(&mut decoder, &eq, ridge);

        // The internal solve is f64 (Cholesky residual ~1e-15), but the returned
        // decoder is f32, so the measurable relative residual bottoms out at the f32
        // floor (~1e-7). Asserting < 1e-6 proves the update CONVERGED to f32 precision
        // — it is not a fixed sweep-count approximation — without chasing a tolerance
        // f32 cannot represent.
        let rel = normal_eq_residual(&eq, &decoder, ridge);
        assert!(
            rel < 1.0e-6,
            "coupled decoder solve must drive ‖(A+ρI)D−B‖/‖B‖ to the f32 floor \
             (< 1e-6), got {rel}"
        );
    }

    #[test]
    fn block_solve_matches_independent_dense_solve() {
        // Exactness cross-check: the connected-component block solve must agree with
        // a single dense Cholesky of the WHOLE assembled (A+ρI) system. (Equivalently,
        // the result has converged — there is no sweep cap that, if raised, would
        // move it.)
        use faer::Side;
        use gam_linalg::faer_ndarray::FaerCholesky;

        let (x, codes, k, p) = overlapping_problem();
        let ridge = 1.0e-6f64;
        let eq = assemble_normal_eq(x.view(), &codes, k, p);

        let mut decoder = Array2::<f32>::zeros((k, p));
        solve_decoder(&mut decoder, &eq, ridge);

        // Dense full system (A + ρI) D = B, solved independently.
        let mut mat = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            mat[[i, i]] = eq.diag[i] + ridge;
        }
        for (&(a, b), &val) in eq.off.iter() {
            mat[[a as usize, b as usize]] = val;
            mat[[b as usize, a as usize]] = val;
        }
        let factor = mat.cholesky(Side::Lower).expect("dense SPD system");
        let dense = factor.solve_mat(&eq.b);

        for i in 0..k {
            for c in 0..p {
                let got = decoder[[i, c]] as f64;
                let want = dense[[i, c]];
                assert!(
                    (got - want).abs() <= 1.0e-5 + 1.0e-5 * want.abs(),
                    "block solve [{i},{c}] = {got} disagrees with dense solve {want}"
                );
            }
        }
    }

    #[test]
    fn returned_ev_is_fresh_code_ev_no_stale_gap() {
        // The convergence-decision EV (= the returned EV) must be the EV of the codes
        // FRESHLY routed against the final normalised decoder — not a stale-code
        // surrogate. We recompute that EV from the public fit's decoder and assert it
        // matches the reported one to f32 rounding.
        let (n, p, k) = (60usize, 6usize, 8usize);
        let mut x = Array2::<f32>::zeros((n, p));
        for i in 0..n {
            for c in 0..p {
                x[[i, c]] = (((i * 3 + c * 7 + 1) % 11) as f32 - 5.0) / 5.0;
            }
        }
        let config = SparseDictConfig {
            n_atoms: k,
            active: 2, // s > 1: exercises the coupled decoder solve
            minibatch: 16,
            max_epochs: 25,
            score_tile: 8,
            code_ridge: 1.0e-6,
            decoder_ridge: 1.0e-6,
            tolerance: 1.0e-9,
        };
        let fit = fit_sparse_dictionary(x.view(), &config).expect("fit");
        let s = fit.active;
        assert!(s > 1, "test must run the coupled s>1 lane");

        let scorer = TileScorer::new(s, config.score_tile);
        let codes = route_and_code_all(
            x.view(),
            fit.decoder.view(),
            &scorer,
            s,
            config.code_ridge,
            config.minibatch,
        );
        let fresh_ev = explained_variance(x.view(), &codes, fit.decoder.view());
        assert!(
            (fresh_ev - fit.explained_variance).abs() < 1.0e-6,
            "returned EV {} must equal fresh-code EV {fresh_ev} (no stale-code gap)",
            fit.explained_variance
        );
    }
}
