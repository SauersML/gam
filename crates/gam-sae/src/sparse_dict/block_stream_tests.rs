//! Tests for the streaming block-sparse lane. The load-bearing one is
//! [`streaming_over_shards_matches_one_shot`]: driving the resumable handle over a
//! corpus's shards must reach the same fixed point (EV, recovered subspaces) as the
//! in-memory [`fit_block_sparse_dictionary`] on the concatenation.

use super::BlockSparseStreamState;
use crate::sparse_dict::{
    BlockSparseConfig, block_gates, block_projections_row, fit_block_sparse_dictionary,
    reconstruct_row, route_row_blocks,
};
use ndarray::{Array2, ArrayView2};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / 2147483648.0) * 2.0 - 1.0
}

/// `K×P` planted orthonormal atoms (distinct eigenvectors of a fixed symmetric
/// matrix), each block spanning a distinct rank-`b` subspace.
fn planted_frames(p: usize, n_blocks: usize, b: usize) -> Array2<f32> {
    use gam_linalg::faer_ndarray::FaerEigh;
    let mut a = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        for j in 0..p {
            a[[i, j]] = ((i * 7 + j * 3 + 1) % 11) as f64 - 5.0;
        }
    }
    let sym = &a + &a.t();
    let (_ev, evecs) = sym.eigh(faer::Side::Lower).expect("orthonormal seed");
    let k = n_blocks * b;
    let mut atoms = Array2::<f32>::zeros((k, p));
    for atom in 0..k {
        let col = evecs.column(atom);
        for c in 0..p {
            atoms[[atom, c]] = col[c] as f32;
        }
    }
    atoms
}

/// Every row lies in exactly ONE planted block's rank-`b` subspace.
fn planted_data(
    planted: &Array2<f32>,
    n_blocks: usize,
    b: usize,
    p: usize,
    n: usize,
) -> Array2<f32> {
    let mut s = 31337u64;
    let mut x = Array2::<f32>::zeros((n, p));
    for i in 0..n {
        let t = i % n_blocks;
        let mut coeffs = vec![0.0f32; b];
        for cf in coeffs.iter_mut() {
            *cf = lcg(&mut s) + 0.5;
        }
        for c in 0..p {
            let mut acc = 0.0f32;
            for (r, &cf) in coeffs.iter().enumerate() {
                acc += cf * planted[[t * b + r, c]];
            }
            x[[i, c]] = acc;
        }
    }
    x
}

/// EV of a block model (frames + γ) over `x` via the public block encode/decode.
fn model_ev(
    x: ArrayView2<'_, f32>,
    decoder: &Array2<f32>,
    gamma: f32,
    g: usize,
    b: usize,
    k: usize,
) -> f64 {
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
    for i in 0..n {
        let row = x.row(i);
        let w = block_projections_row(row, decoder.view(), g, b);
        let gates = block_gates(w.view());
        let sel: Vec<u32> = route_row_blocks(&gates, k)
            .iter()
            .map(|&(gg, _)| gg)
            .collect();
        let recon = reconstruct_row(row, decoder.view(), &sel, gamma, b);
        for c in 0..p {
            let r = x[[i, c]] as f64 - recon[c] as f64;
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

fn config(g: usize, b: usize, k: usize) -> BlockSparseConfig {
    BlockSparseConfig {
        n_blocks: g,
        block_size: b,
        block_topk: k,
        max_epochs: 80,
        minibatch: 64,
        block_tile: 8,
        frame_ridge: 1.0e-9,
        aux_k: g,
        tolerance: 1.0e-10,
    }
}

#[test]
fn streaming_over_shards_matches_one_shot() {
    let (p, b, g) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, g, b);
    let x = planted_data(&planted, g, b, p, 180);
    let cfg = config(g, b, 1);

    let one_shot = fit_block_sparse_dictionary(x.view(), &cfg).expect("one-shot block fit");

    // Four contiguous shards whose concatenation (row order) is exactly `x`.
    let n = x.nrows();
    let chunk = n / 4;
    let shards: Vec<ArrayView2<'_, f32>> = (0..4)
        .map(|i| {
            let start = i * chunk;
            let end = if i == 3 { n } else { start + chunk };
            x.slice(ndarray::s![start..end, ..])
        })
        .collect();

    let mut state = BlockSparseStreamState::new(x.view(), &cfg).expect("fit_begin");
    for _ in 0..cfg.max_epochs {
        for shard in &shards {
            state.partial_fit(*shard).expect("partial_fit");
        }
        let stats = state.end_epoch().expect("end_epoch");
        if stats.converged {
            break;
        }
    }
    let art = state.finalize();

    assert_eq!(
        art.decoder.shape(),
        one_shot.decoder.shape(),
        "streamed frames must have the one-shot shape"
    );
    let ev_stream = model_ev(x.view(), &art.decoder, art.gamma, g, b, art.block_topk);
    assert!(
        ev_stream > 0.9,
        "streamed block fit should reconstruct the planted subspaces well, EV={ev_stream}"
    );
    assert!(
        (ev_stream - one_shot.explained_variance).abs() < 0.1,
        "streamed EV {ev_stream} must track one-shot EV {}",
        one_shot.explained_variance
    );
}

#[test]
fn warm_start_persists_across_epochs() {
    // A later epoch's pre-refresh EV (which sees frames refreshed by earlier epochs)
    // must improve on the first epoch's — the frames/γ warm-start across calls.
    let (p, b, g) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, g, b);
    let x = planted_data(&planted, g, b, p, 150);
    let mut cfg = config(g, b, 1);
    cfg.max_epochs = 6;

    let mut state = BlockSparseStreamState::new(x.view(), &cfg).expect("fit_begin");
    let mut evs = Vec::new();
    for _ in 0..cfg.max_epochs {
        state.partial_fit(x.view()).expect("partial_fit");
        evs.push(state.end_epoch().expect("end_epoch").explained_variance);
    }
    assert!(
        evs[evs.len() - 1] > evs[0] + 1.0e-4,
        "later-epoch EV {} must improve on first-epoch EV {} (warm-start persisted)",
        evs[evs.len() - 1],
        evs[0]
    );
}

#[test]
fn revival_reseeds_dead_block_from_worst_residual_row() {
    // Seed spans only the first two planted subspaces; the third block starts dead.
    // Streaming rows that live in the third subspace makes those rows the worst-
    // reconstructed, and AuxK revival must reseed the dead block onto their residual
    // (recovering the third subspace), not leave it dead.
    let (p, b, g) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, g, b);
    let x = planted_data(&planted, g, b, p, 150);
    let cfg = config(g, b, 1);

    let mut state = BlockSparseStreamState::new(x.view(), &cfg).expect("fit_begin");
    let mut saw_dead = false;
    let mut saw_revive = false;
    for _ in 0..cfg.max_epochs {
        state.partial_fit(x.view()).expect("partial_fit");
        let stats = state.end_epoch().expect("end_epoch");
        saw_dead |= stats.dead > 0;
        saw_revive |= stats.revived > 0;
        if stats.converged {
            break;
        }
    }
    let art = state.finalize();
    // Every planted subspace is used: no block ends with zero utilisation.
    let live = art.block_utilization.iter().filter(|&&u| u > 0.0).count();
    assert_eq!(
        live, g,
        "all {g} blocks must be live after revival (util>0)"
    );
    let ev = model_ev(x.view(), &art.decoder, art.gamma, g, b, art.block_topk);
    assert!(
        ev > 0.9,
        "revival should let the fit reach all planted subspaces, EV={ev}"
    );
    // Revival machinery actually engaged at some point (dictionary started under-
    // populated and AuxK filled it).
    assert!(
        saw_dead,
        "dictionary must pass through a dead-block state before revival"
    );
    assert!(
        saw_revive,
        "AuxK revival must engage on the under-populated dictionary"
    );
}
