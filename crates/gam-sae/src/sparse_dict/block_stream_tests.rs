//! Tests for the streaming block-sparse lane. The load-bearing one is
//! [`streaming_over_shards_matches_one_shot`]: driving the resumable handle over a
//! corpus's shards must reach the same fixed point (EV, recovered subspaces) as the
//! in-memory [`fit_block_sparse_dictionary`] on the concatenation.

use super::BlockSparseStreamState;
use super::test_support::BlockStreamTestAccess;
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
        matryoshka_prefix: false,
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
    let art = state.finalize().expect("finalize");

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
    // The second epoch's pre-refresh EV must be exactly the reconstruction emitted
    // by the first epoch's refreshed frames/γ: partial_fit consumes the prior state
    // instead of silently reseeding at the call boundary.
    let (p, b, g) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, g, b);
    let x = planted_data(&planted, g, b, p, 150);
    let mut cfg = config(g, b, 1);
    cfg.max_epochs = 2;
    cfg.aux_k = 0;

    // Tilt the seed away from the planted subspaces so the first refresh is
    // observably different from initialization; a reset cannot pass vacuously.
    let seed = Array2::<f32>::from_shape_fn(x.raw_dim(), |(row, col)| {
        x[[row, col]] + 0.2 * ((row + 3 * col) as f32 * 0.11).sin()
    });

    let mut state = BlockSparseStreamState::new(seed.view(), &cfg).expect("fit_begin");
    let (initial_decoder, initial_gamma) = state.model_snapshot_for_test();
    state.partial_fit(x.view()).expect("first partial_fit");
    state.end_epoch().expect("first end_epoch");
    let (warm_decoder, warm_gamma) = state.model_snapshot_for_test();

    let decoder_change = initial_decoder
        .iter()
        .zip(warm_decoder.iter())
        .map(|(&before, &after)| {
            let delta = f64::from(after - before);
            delta * delta
        })
        .sum::<f64>()
        .sqrt();
    assert!(
        decoder_change > 1.0e-4 || f64::from(warm_gamma - initial_gamma).abs() > 1.0e-4,
        "first epoch must materially refresh decoder/gamma so the handoff check is non-vacuous"
    );

    let expected_second_ev = model_ev(x.view(), &warm_decoder, warm_gamma, g, b, cfg.block_topk);
    state.partial_fit(x.view()).expect("second partial_fit");
    let second = state.end_epoch().expect("second end_epoch");

    assert!(
        (second.explained_variance - expected_second_ev).abs() <= 1.0e-10,
        "second pass EV {} must use the exact first-epoch decoder/gamma (expected {})",
        second.explained_variance,
        expected_second_ev
    );
}

#[test]
fn evidence_birth_uses_worst_residual_row() {
    // Seed spans only the first two planted subspaces; the third block starts dead.
    // Streaming rows that live in the third subspace makes those rows the worst-
    // reconstructed, and the AuxK proposal must birth the dead block from their
    // residual after exact evidence admission (recovering the third subspace).
    let (p, b, g) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, g, b);
    let x = planted_data(&planted, g, b, p, 150);
    let cfg = config(g, b, 1);

    // The seed must genuinely omit the third subspace, otherwise farthest-point
    // seeding ([`seed_decoder`]) spreads an atom into every block and no block ever
    // starts dead. `planted_data` places row `i` in subspace `i % g`, so the rows
    // with `i % g != g - 1` live entirely in the first two subspaces; block 2's
    // seeded frame then lands inside span{subspace 0, subspace 1}, orthogonal to the
    // (later-streamed) subspace-2 rows, so it routes to zero usage and its
    // residual-row birth is accepted by the exact next-pass comparison.
    let seed_rows: Vec<usize> = (0..x.nrows()).filter(|&i| i % g != g - 1).collect();
    let mut seed = Array2::<f32>::zeros((seed_rows.len(), p));
    for (dst, &src) in seed_rows.iter().enumerate() {
        seed.row_mut(dst).assign(&x.row(src));
    }

    let mut state = BlockSparseStreamState::new(seed.view(), &cfg).expect("fit_begin");
    // Force the last block DEAD deterministically (its routing gate is then exactly 0
    // for every row) so AuxK birth is exercised without relying on routing round-off
    // (a parallel-reduction change, #49c27a883, could otherwise bootstrap it).
    state.zero_block_for_test(g - 1);
    let mut saw_dead = false;
    let mut saw_accepted_birth = false;
    for _ in 0..cfg.max_epochs {
        state.partial_fit(x.view()).expect("partial_fit");
        let stats = state.end_epoch().expect("end_epoch");
        saw_dead |= stats.dead > 0;
        saw_accepted_birth |= stats.accepted_births > 0;
        if stats.converged {
            break;
        }
    }
    let art = state.finalize().expect("finalize");
    // Every planted subspace is used: no block ends with zero utilisation.
    let live = art.block_utilization.iter().filter(|&&u| u > 0.0).count();
    assert_eq!(
        live, g,
        "all {g} blocks must be live after evidence-admitted birth (util>0)"
    );
    let ev = model_ev(x.view(), &art.decoder, art.gamma, g, b, art.block_topk);
    assert!(
        ev > 0.9,
        "the admitted birth should let the fit reach all planted subspaces, EV={ev}"
    );
    // Revival machinery actually engaged at some point (dictionary started under-
    // populated and AuxK filled it).
    assert!(
        saw_dead,
        "dictionary must pass through a dead-block state before birth"
    );
    assert!(
        saw_accepted_birth,
        "the residual-row proposal must commit after strict full-pass improvement"
    );
}

#[test]
fn overcomplete_stream_accepts_one_evidence_birth_then_dead_tail_is_quiescent_2023() {
    // Rank-2 data with G=16 reproduces the K≫intrinsic-rank boundary behind
    // #2023. Block 0 starts on e0 and every other frame is dead. Exactly one e1
    // residual birth is warranted; after it commits the remaining fourteen dead
    // blocks must stay quiescent so the stream can certify instead of reseeding
    // them forever.
    let (rows, p, g, b) = (64usize, 2usize, 16usize, 1usize);
    let x = Array2::<f32>::from_shape_fn((rows, p), |(row, column)| {
        if column == row % 2 { 1.0 } else { 0.0 }
    });
    let mut decoder = Array2::<f32>::zeros((g * b, p));
    decoder[[0, 0]] = 1.0;
    let cfg = BlockSparseConfig {
        n_blocks: g,
        block_size: b,
        block_topk: 1,
        max_epochs: 8,
        minibatch: rows,
        block_tile: g,
        frame_ridge: 0.0,
        aux_k: g,
        matryoshka_prefix: false,
        tolerance: 0.0,
    };
    let mut state =
        BlockSparseStreamState::new_with_decoder(decoder, &cfg).expect("stream state");
    let mut accepted_total = 0usize;
    let mut saw_pending = false;
    let mut final_stats = None;
    for _ in 0..cfg.max_epochs {
        state.partial_fit(x.view()).expect("stream rank-2 corpus");
        let stats = state.end_epoch().expect("close rank-2 epoch");
        accepted_total += stats.accepted_births;
        saw_pending |= stats.birth_pending;
        final_stats = Some(stats);
        if stats.converged {
            break;
        }
    }
    let final_stats = final_stats.expect("at least one epoch");
    assert!(saw_pending, "a residual-row birth must be staged for e1");
    assert_eq!(
        accepted_total, 1,
        "only the missing rank-1 direction has positive exact evidence"
    );
    assert!(final_stats.converged, "dead tail prevented certification");
    assert!(!final_stats.birth_pending);
    assert_eq!(final_stats.dead, g - 2);

    let artifact = state.finalize().expect("quiescent overcomplete artifact");
    assert_eq!(
        artifact
            .block_utilization
            .iter()
            .filter(|&&value| value > 0.0)
            .count(),
        2,
    );
    assert!((artifact.explained_variance - 1.0).abs() <= f64::EPSILON);
}
