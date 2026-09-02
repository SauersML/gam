//! Tests for the streaming block-sparse lane. The load-bearing one is
//! [`streaming_over_shards_matches_one_shot`]: driving the resumable handle over a
//! corpus's shards must reach the same fixed point (EV, recovered subspaces) as the
//! in-memory [`fit_block_sparse_dictionary`] on the concatenation.

use super::BlockSparseStreamState;
use crate::sparse_dict::BlockSparseConfig;
use ndarray::Array2;

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
