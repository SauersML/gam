//! Alternating minibatched trainer: route → sparse codes → decoder refresh →
//! unit-norm projection. No dense `N×K` object is ever formed.
//!
//! The decoder refresh is the **method of optimal directions** (MOD) restricted
//! to the sparse support. With codes fixed, the reconstruction loss
//! `Σ_i ‖x_i − Σ_j c_{ij} d_{a_{ij}}‖²` is quadratic in the decoder `D` and its
//! normal equations are `D (CᵀC + ρI) = CᵀX`, where `C` is the (sparse, never
//! materialised) `N×K` code matrix. We accumulate `A = CᵀC` (`K×K`, but only
//! the few entries touched by co-active atoms are non-zero) and `B = CᵀX`
//! (`K×P`) by streaming minibatches, then solve atom-by-atom. For a clean
//! `top_s = 1` lane `A` is diagonal and the refresh is a per-atom rescaled mean
//! — exactly MOD / k-SVD's dictionary step. We solve the general `s > 1` case by
//! a Jacobi sweep over the sparse coupling, which keeps the cost off `K²`.

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
        let active_lists = scorer.route_minibatch(block, decoder);
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

    for epoch in 0..config.max_epochs {
        epochs_run = epoch + 1;

        // (a)+(b) route + sparse codes for every row, in minibatches: each
        // minibatch is routed by one batched GEMM per column tile (peak score
        // working set `minibatch × score_tile`, never `N × K`) and its per-row
        // active-set code solves run in parallel.
        let codes = route_and_code_all(
            x,
            decoder.view(),
            &scorer,
            s,
            config.code_ridge,
            config.minibatch,
        );

        // (c) decoder refresh from the sparse normal equations, streamed by
        // minibatch into the per-atom accumulators.
        refresh_decoder(x, &codes, &mut decoder, k, p, config);

        // (d) unit-norm projection (identifies code scale).
        unit_norm_rows(&mut decoder);

        let ev = explained_variance(x, &codes, decoder.view());
        let improve = ev - prev_ev;
        if improve.abs() <= config.tolerance && epoch > 0 {
            converged = true;
            break;
        }
        prev_ev = ev;
    }

    // Re-route once against the final, unit-normed decoder so the stored codes
    // match the returned dictionary exactly.
    let codes = route_and_code_all(
        x,
        decoder.view(),
        &scorer,
        s,
        config.code_ridge,
        config.minibatch,
    );
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

/// Refresh the decoder from the sparse normal equations `D(A+ρI)=B`,
/// `A = CᵀC`, `B = CᵀX`, where the code matrix `C` is never materialised. Only
/// atom pairs that co-fire in some row appear in `A`, so the coupling is sparse
/// and we solve by a damped Jacobi sweep that touches only those couplings.
fn refresh_decoder(
    x: ArrayView2<'_, f32>,
    codes: &[SparseCode],
    decoder: &mut Array2<f32>,
    k: usize,
    p: usize,
    config: &SparseDictConfig,
) {
    let ridge = config.decoder_ridge as f64;
    // Diagonal of A: Σ_i c_{ik}².
    let mut diag = vec![0.0f64; k];
    // B = CᵀX, K×P, accumulated sparsely.
    let mut b = Array2::<f64>::zeros((k, p));
    // Off-diagonal couplings A_{kl} (k<l) keyed by (k,l).
    let mut off: HashMap<(u32, u32), f64> = HashMap::new();

    for (row_idx, code) in codes.iter().enumerate() {
        let xi = x.row(row_idx);
        for a in 0..code.indices.len() {
            let ca = code.codes[a] as f64;
            if ca == 0.0 {
                continue;
            }
            let ka = code.indices[a];
            diag[ka as usize] += ca * ca;
            let brow = ka as usize;
            for c in 0..p {
                b[[brow, c]] += ca * xi[c] as f64;
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

    // Build adjacency for the Jacobi sweep: for each atom, its coupled
    // (neighbour, A_{kl}) pairs.
    let mut neigh: Vec<Vec<(u32, f64)>> = vec![Vec::new(); k];
    for (&(ka, kb), &val) in off.iter() {
        neigh[ka as usize].push((kb, val));
        neigh[kb as usize].push((ka, val));
    }

    // Damped Jacobi: D_k ← (B_k − Σ_{l≠k} A_{kl} D_l) / (A_kk + ρ). A few
    // sweeps suffice because the coupling is weak (atoms rarely co-fire); atoms
    // with no code keep their seeded direction.
    let new_decoder_from = |decoder: &Array2<f32>| -> Array2<f32> {
        let mut out = decoder.clone();
        for atom in 0..k {
            let denom = diag[atom] + ridge;
            if denom <= 1.0e-12 {
                // Dead atom this epoch: leave its seeded direction in place so a
                // later epoch can route rows to it (no permanent collapse).
                continue;
            }
            for c in 0..p {
                let mut acc = b[[atom, c]];
                for &(nb, aval) in &neigh[atom] {
                    acc -= aval * decoder[[nb as usize, c]] as f64;
                }
                out[[atom, c]] = (acc / denom) as f32;
            }
        }
        out
    };

    let sweeps = if off.is_empty() { 1 } else { 4 };
    for _ in 0..sweeps {
        *decoder = new_decoder_from(decoder);
    }
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
