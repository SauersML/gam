//! Tests for the streaming block-sparse lane. The load-bearing one is
//! [`streaming_over_shards_matches_one_shot`]: driving the resumable handle over a
//! corpus's shards must reach the same fixed point (EV, recovered subspaces) as the
//! in-memory [`fit_block_sparse_dictionary`] on the concatenation.

use super::BlockSparseStreamState;
use crate::sparse_dict::BlockSparseConfig;
use ndarray::Array2;

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
fn overcomplete_stream_accepts_one_evidence_birth_then_dead_tail_is_quiescent_2023() {
    // Rank-2 data with G=16 reproduces the K≫intrinsic-rank boundary behind
    // #2023. Block 0 starts on e0 and every other frame is dead. Exactly one e1
    // residual birth is warranted; after it commits the remaining fourteen dead
    // blocks must stay quiescent so the stream can certify instead of reseeding
    // them forever.
    let (rows, p, g, b) = (64usize, 2usize, 16usize, 1usize);
    let x = Array2::<f32>::from_shape_fn(
        (rows, p),
        |(row, column)| {
            if column == row % 2 { 1.0 } else { 0.0 }
        },
    );
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
    let mut state = BlockSparseStreamState::new_with_decoder(decoder, &cfg).expect("stream state");
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
