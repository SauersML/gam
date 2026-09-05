//! Streaming block updates must use one coordinated gamma/frame state, descend
//! in their conditional objective, and certify the model actually returned.

use super::BlockSparseStreamState;
use crate::sparse_dict::BlockSparseConfig;
use ndarray::{Array2, array};

fn coupled_fixture() -> (Array2<f32>, Array2<f32>, BlockSparseConfig) {
    let x = array![[1.0_f32, 2.0], [-2.0, 1.0], [0.5, 0.2], [-0.7, -0.4]];
    let decoder = array![[1.0_f32, 0.0], [0.6, 0.8]];
    let config = BlockSparseConfig {
        n_blocks: 2,
        block_size: 1,
        block_topk: 2,
        max_epochs: 64,
        minibatch: 4,
        block_tile: 2,
        frame_ridge: 0.0,
        aux_k: 0,
        matryoshka_prefix: false,
        tolerance: 1e-6,
    };
    (x, decoder, config)
}

#[test]
fn parallel_stream_moments_match_dense_reference_across_batches_and_shards() {
    let x = Array2::from_shape_fn((17, 4), |(row, feature)| {
        if row == 0 {
            0.0
        } else {
            ((row * 7 + feature * 13) as f32 * 0.17).sin()
        }
    });
    let decoder = array![
        [1.0_f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.6, 0.0, 0.0, 0.8],
        [0.0, 0.0, 1.0, 0.0],
    ];
    // Independent dense frozen-code algebra. All three blocks are selected on
    // nonzero rows; the zero row exercises padded slots without phantom usage.
    let weights = x.mapv(f64::from).dot(&decoder.mapv(f64::from).t());
    let total = weights.dot(&decoder.mapv(f64::from));
    let gamma = 0.37_f32;
    let baseline_gamma = 0.61_f32;
    let residual = x.mapv(f64::from) - &total * gamma as f64;
    let baseline_residual = x.mapv(f64::from) - &total * baseline_gamma as f64;
    let expected_rss = residual.iter().map(|v| v * v).sum::<f64>();
    let expected_baseline_rss = baseline_residual.iter().map(|v| v * v).sum::<f64>();
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for batch in [1, 5, 17] {
                for shard_rows in [3, 17] {
                    let mut config = BlockSparseConfig::new(3, 2);
                    config.block_topk = 3;
                    config.minibatch = batch;
                    config.aux_k = 2;
                    let mut state =
                        BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
                    state.gamma = gamma;
                    state.pending_birth = Some(super::PendingBlockBirth {
                        block: 0,
                        baseline_decoder: decoder.clone(),
                        baseline_gamma,
                        baseline_rss: 0.0,
                        baseline_rows: 0,
                        baseline_usage: vec![0; 3],
                        baseline_second: (0..3).map(|_| Array2::zeros((2, 2))).collect(),
                    });
                    for shard in x.axis_chunks_iter(ndarray::Axis(0), shard_rows) {
                        state.partial_fit(shard).unwrap();
                    }
                    assert_eq!(state.row_count, x.nrows());
                    assert_eq!(state.usage, vec![16; 3]);
                    assert_eq!(state.alive_count, 3);
                    assert!((state.rss - expected_rss).abs() < 1e-11);
                    let num = x
                        .iter()
                        .zip(total.iter())
                        .map(|(&x, &v)| x as f64 * v)
                        .sum::<f64>();
                    let den = total.iter().map(|v| v * v).sum::<f64>();
                    assert!((state.gamma_num - num).abs() < 1e-11);
                    assert!((state.gamma_den - den).abs() < 1e-11);
                    let pending = state.pending_birth.as_ref().unwrap();
                    assert_eq!(pending.baseline_rows, x.nrows());
                    assert_eq!(pending.baseline_usage, state.usage);
                    assert!((pending.baseline_rss - expected_baseline_rss).abs() < 1e-11);
                    for block in 0..3 {
                        let w = weights.slice(ndarray::s![.., block * 2..(block + 1) * 2]);
                        let own = w.dot(
                            &decoder
                                .slice(ndarray::s![block * 2..(block + 1) * 2, ..])
                                .mapv(f64::from),
                        );
                        let expected_cross = (&total - &own).t().dot(&w);
                        let expected_data = x.mapv(f64::from).t().dot(&w);
                        let expected_second = w.t().dot(&w);
                        for (got, expected) in [
                            (&state.cross[block], expected_cross),
                            (&state.data_cross[block], expected_data),
                            (&state.second[block], expected_second.clone()),
                            (
                                &pending.baseline_second[block],
                                expected_second * (baseline_gamma as f64).powi(2),
                            ),
                        ] {
                            assert!(
                                got.iter()
                                    .zip(expected.iter())
                                    .all(|(a, b)| (a - b).abs() < 1e-11)
                            );
                        }
                    }
                    for feature in 0..4 {
                        let column = x.column(feature);
                        assert!(
                            (state.col_sum[feature]
                                - column.iter().map(|&v| v as f64).sum::<f64>())
                            .abs()
                                < 1e-11
                        );
                        assert!(
                            (state.col_sumsq[feature]
                                - column.iter().map(|&v| (v as f64).powi(2)).sum::<f64>())
                            .abs()
                                < 1e-11
                        );
                    }
                    let mut worst: Vec<usize> = (0..x.nrows()).collect();
                    worst.sort_by(|&a, &b| {
                        let norm = |row| residual.row(row).iter().map(|v| v * v).sum::<f64>();
                        norm(b).total_cmp(&norm(a)).then(a.cmp(&b))
                    });
                    let ranked = state.reservoir.ranked();
                    assert_eq!(ranked.len(), 4);
                    for (entry, &row) in ranked.iter().zip(&worst) {
                        assert_eq!(entry.global_index, row as u64);
                        assert!(
                            entry
                                .residual
                                .iter()
                                .zip(residual.row(row).iter())
                                .all(|(&a, &b)| (a as f64 - b).abs() < 1e-6)
                        );
                    }
                }
            }
        });
    }
}

#[test]
fn frame_refresh_uses_the_new_gamma_and_is_invariant_to_its_initial_value_2825() {
    let (x, decoder, config) = coupled_fixture();
    for orientation in [-1.0, 1.0] {
        let decoder = decoder.mapv(|value| orientation * value);
        let mut reference = None;
        for gamma in [0.2, 1.0, 2.0] {
            let mut state =
                BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
            state.gamma = gamma;
            for row in x.outer_iter() {
                state
                    .partial_fit(row.insert_axis(ndarray::Axis(0)))
                    .unwrap();
            }
            let stats = state.end_epoch().unwrap();
            assert!(!stats.converged);
            if let Some((previous_decoder, previous_gamma)) = &reference {
                assert_eq!(&state.decoder, previous_decoder);
                assert_eq!(&stats.gamma, previous_gamma);
            }
            reference = Some((state.decoder.clone(), stats.gamma));
            // Price the candidate with the frozen codes, independently of the
            // stream's moment algebra. Simultaneous updates must decrease this loss.
            let loss = |candidate: &Array2<f32>| -> f64 {
                x.outer_iter()
                    .map(|row| {
                        let weights: Vec<f64> = decoder
                            .outer_iter()
                            .map(|direction| {
                                direction
                                    .iter()
                                    .zip(row.iter())
                                    .map(|(&d, &x)| d as f64 * x as f64)
                                    .sum()
                            })
                            .collect();
                        (0..x.ncols())
                            .map(|feature| {
                                let reconstruction: f64 = (0..2)
                                    .map(|block| {
                                        stats.gamma as f64
                                            * weights[block]
                                            * candidate[[block, feature]] as f64
                                    })
                                    .sum();
                                (row[feature] as f64 - reconstruction).powi(2)
                            })
                            .sum::<f64>()
                    })
                    .sum()
            };
            assert!(loss(&state.decoder) < loss(&decoder));
        }
    }
}

#[test]
fn parallel_stream_rejects_selected_duplicate_birth_using_complete_baseline() {
    let x = Array2::from_shape_fn(
        (17, 2),
        |(_, column)| if column == 0 { 1.0_f32 } else { 0.0 },
    );
    let baseline = array![[0.0_f32, 0.0], [1.0, 0.0]];
    let candidate = array![[1.0_f32, 0.0], [1.0, 0.0]];
    let mut config = BlockSparseConfig::new(2, 1);
    config.block_topk = 1;
    config.minibatch = 5;
    config.aux_k = 1;
    config.frame_ridge = 0.0;
    let mut state = BlockSparseStreamState::new_with_decoder(candidate, &config).unwrap();
    state.pending_birth = Some(super::PendingBlockBirth {
        block: 0,
        baseline_decoder: baseline.clone(),
        baseline_gamma: 1.0,
        baseline_rss: 0.0,
        baseline_rows: 0,
        baseline_usage: vec![0; 2],
        baseline_second: (0..2).map(|_| Array2::zeros((1, 1))).collect(),
    });
    state.partial_fit(x.view()).unwrap();
    assert_eq!(
        state.usage,
        vec![17, 0],
        "the duplicate must win the routing tie"
    );
    let pending = state.pending_birth.as_ref().unwrap();
    assert_eq!(pending.baseline_usage, vec![0, 17]);
    assert_eq!(pending.baseline_rss, state.rss);
    let stats = state.end_epoch().unwrap();
    assert_eq!(stats.accepted_births, 0);
    assert!(
        !stats.converged,
        "rejection requires a measured baseline pass"
    );
    assert_eq!(state.decoder, baseline);
    assert_eq!(state.last_usage, vec![0, 17]);
    assert_eq!(state.last_second[1][[0, 0]], 17.0);
}

#[test]
fn equal_ev_cannot_certify_changing_gamma_or_frames_2825() {
    let (x, decoder, config) = coupled_fixture();
    let mut first = BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
    first.partial_fit(x.view()).unwrap();
    let measured = first.end_epoch().unwrap();
    let mut state = BlockSparseStreamState::new_with_decoder(decoder, &config).unwrap();
    state.prev_ev = measured.explained_variance;
    state.epochs_run = 1;
    state.partial_fit(x.view()).unwrap();
    let stats = state.end_epoch().unwrap();
    assert_eq!(stats.explained_variance, measured.explained_variance);
    assert!(stats.gamma_residual > config.tolerance);
    assert!(stats.frame_residual > config.tolerance);
    assert!(!stats.converged);
    assert!(state.finalize().is_err());
}

#[test]
fn coupled_stream_certifies_the_returned_frames_gamma_and_fresh_ev_2825() {
    let (x, decoder, config) = coupled_fixture();
    let mut state = BlockSparseStreamState::new_with_decoder(decoder, &config).unwrap();
    let mut last = None;
    for _ in 0..config.max_epochs {
        state.partial_fit(x.view()).unwrap();
        let stats = state.end_epoch().unwrap();
        last = Some(stats);
        if stats.converged {
            break;
        }
    }
    let last = last.unwrap();
    assert!(last.converged, "{last:?}");
    assert!(last.gamma_residual <= config.tolerance);
    assert!(last.frame_residual <= config.tolerance);
    let artifact = state.finalize().unwrap();
    let mean = x.mean_axis(ndarray::Axis(0)).unwrap();
    let mut rss = 0.0;
    let mut tss = 0.0;
    for row in x.outer_iter() {
        let mut reconstructed = vec![0.0_f64; x.ncols()];
        for direction in artifact.decoder.outer_iter() {
            let projection: f64 = direction
                .iter()
                .zip(row.iter())
                .map(|(&d, &x)| d as f64 * x as f64)
                .sum();
            for (feature, value) in reconstructed.iter_mut().enumerate() {
                *value += artifact.gamma as f64 * projection * direction[feature] as f64;
            }
        }
        for feature in 0..x.ncols() {
            rss += (row[feature] as f64 - reconstructed[feature]).powi(2);
            tss += (row[feature] as f64 - mean[feature] as f64).powi(2);
        }
    }
    assert!((artifact.explained_variance - (1.0 - rss / tss)).abs() < 1e-6);
    assert!(artifact.explained_variance > 0.99);
    state.partial_fit(x.view()).unwrap();
    assert!(
        state.finalize().is_err(),
        "a new unclosed pass invalidates the certificate"
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
