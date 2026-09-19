//! Streaming block updates must use one coordinated gamma/frame state, descend
//! in their conditional objective, and certify the model actually returned.

use super::BlockSparseStreamState;
use crate::sparse_dict::BlockSparseConfig;
use ndarray::{Array2, array};

#[test]
fn streaming_seed_is_data_placed_when_dictionary_is_overcomplete_2023() {
    let direction = [1.0_f32, 2.0, 3.0, 4.0];
    let seed = Array2::from_shape_fn((32, direction.len()), |(row, column)| {
        (row as f32 - 15.5) * direction[column]
    });
    let config = BlockSparseConfig::new(16, 1);
    let state = BlockSparseStreamState::new(seed.view(), &config).expect("valid streaming seed");
    let direction_norm = direction
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    for block in 0..config.n_blocks {
        let alignment = state
            .decoder
            .row(block)
            .iter()
            .zip(direction.iter())
            .map(|(&left, &right)| left * right / direction_norm)
            .sum::<f32>()
            .abs();
        assert!((alignment - 1.0).abs() <= 8.0 * f32::EPSILON);
    }
}

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
fn frame_proposals_and_certificates_are_identical_across_worker_counts() {
    let mut seed = 2826_u64;
    let mut sample = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let x = Array2::from_shape_fn((37, 8), |_| sample());
    let mut decoder = Array2::from_shape_fn((12, 8), |_| sample());
    for block in 0..5 {
        let mut frame = decoder
            .slice(ndarray::s![block * 2..(block + 1) * 2, ..])
            .to_owned();
        crate::sparse_dict::block::gram_schmidt_rows(&mut frame);
        decoder
            .slice_mut(ndarray::s![block * 2..(block + 1) * 2, ..])
            .assign(&frame);
    }
    decoder.slice_mut(ndarray::s![10.., ..]).fill(0.0);
    let mut config = BlockSparseConfig::new(6, 2);
    config.block_topk = 3;
    config.minibatch = 13;
    config.aux_k = 0;
    let mut reference = None;
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let observed = pool.install(|| {
            let mut state =
                BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
            let mut epochs = Vec::new();
            for _ in 0..3 {
                state.partial_fit(x.view()).unwrap();
                let stats = state.end_epoch().unwrap();
                assert!(
                    state
                        .decoder
                        .slice(ndarray::s![10.., ..])
                        .iter()
                        .all(|&v| v == 0.0)
                );
                assert!(stats.frame_residual.is_finite());
                epochs.push((
                    state.decoder.clone(),
                    stats.gamma,
                    stats.explained_variance,
                    stats.gamma_residual,
                    stats.frame_residual,
                    stats.converged,
                    state.last_second.clone(),
                ));
            }
            epochs
        });
        if let Some(expected) = reference.as_ref() {
            assert_eq!(
                &observed, expected,
                "worker count changed the proposed model or certificate"
            );
        } else {
            reference = Some(observed);
        }
    }
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
    // Independent dense projector algebra. All three blocks are selected on
    // nonzero rows; the zero row exercises padded slots without phantom usage.
    // Each block projects through the orthogonal projector of its stored rows,
    // `Uᵀ(UUᵀ)⁻¹U` (#2502): its span coordinates are its inner products `x Uᵀ` times
    // the closed-form inverse of its 2×2 Gram. The f32 row [0.6, 0, 0, 0.8] is not
    // exactly unit, so `UᵀU` would price a different reconstruction.
    let decoder64 = decoder.mapv(f64::from);
    let mut weights = x.mapv(f64::from).dot(&decoder64.t());
    for block in 0..3 {
        let frame = decoder64.slice(ndarray::s![block * 2..(block + 1) * 2, ..]);
        let gram = frame.dot(&frame.t());
        let det = gram[[0, 0]] * gram[[1, 1]] - gram[[0, 1]] * gram[[1, 0]];
        let inverse = array![
            [gram[[1, 1]] / det, -gram[[0, 1]] / det],
            [-gram[[1, 0]] / det, gram[[0, 0]] / det]
        ];
        let mut coordinates = weights.slice_mut(ndarray::s![.., block * 2..(block + 1) * 2]);
        let solved = coordinates.dot(&inverse);
        coordinates.assign(&solved);
    }
    let total = weights.dot(&decoder64);
    let gamma = 0.37_f32;
    let baseline_gamma = 0.61_f32;
    let residual = x.mapv(f64::from) - &total * gamma as f64;
    let baseline_residual = x.mapv(f64::from) - &total * baseline_gamma as f64;
    let expected_rss = residual.iter().map(|v| v * v).sum::<f64>();
    let expected_baseline_rss = baseline_residual.iter().map(|v| v * v).sum::<f64>();
    // Nonzero search rows for every block, so the accumulated operator is checked
    // on the search columns as well as on the frame columns.
    let searched = Array2::from_shape_fn((12, 4), |(row, feature)| {
        ((row * 3 + feature * 5 + 1) as f32 * 0.29).cos()
    });
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
                    state.search_directions = searched.clone();
                    state.pending_birth = Some(super::PendingBlockBirth {
                        block: 0,
                        baseline_decoder: decoder.clone(),
                        baseline_gamma,
                        baseline_rss: 0.0,
                        baseline_rows: 0,
                        baseline_usage: vec![0; 3],
                        baseline_second: (0..3).map(|_| Array2::zeros((2, 2))).collect(),
                        baseline_supports: Vec::new(),
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
                        let x64 = x.mapv(f64::from);
                        // A row's majorizer weight for each block it admits is its
                        // admitted count. Every nonzero row here admits all three
                        // blocks, so the weight is 3 on those rows.
                        let code_norm = |row: usize, h: usize| {
                            let code = weights.slice(ndarray::s![row, h * 2..(h + 1) * 2]);
                            code.dot(&code).sqrt()
                        };
                        let majorizer = ndarray::Array1::from_shape_fn(x.nrows(), |row| {
                            if code_norm(row, block) == 0.0 {
                                return 0.0;
                            }
                            (0..3).filter(|&h| code_norm(row, h) != 0.0).count() as f64
                        });
                        let weighted = majorizer.view().insert_axis(ndarray::Axis(1));
                        let v = &own * &weighted - &total;
                        let subspace = ndarray::concatenate(
                            ndarray::Axis(0),
                            &[
                                decoder
                                    .slice(ndarray::s![block * 2..(block + 1) * 2, ..])
                                    .mapv(f64::from)
                                    .view(),
                                searched
                                    .slice(ndarray::s![block * 4..(block + 1) * 4, ..])
                                    .mapv(f64::from)
                                    .view(),
                            ],
                        )
                        .unwrap();
                        let expected_coupling = (x64.t().dot(&v) + v.t().dot(&x64)
                            - x64.t().dot(&(&x64 * &weighted)))
                        .dot(&subspace.t());
                        let expected_data = x64.t().dot(&x64).dot(&subspace.t());
                        let expected_second = w.t().dot(&w);
                        let expected_normal = w.t().dot(&(&w * &(&weighted - 1.0)));
                        for (got, expected) in [
                            (&state.coupling[block], expected_coupling),
                            (&state.data_cross[block], expected_data),
                            (&state.second[block], expected_second.clone()),
                            (&state.normal_second[block], expected_normal),
                            (
                                &pending.baseline_second[block],
                                expected_second * (baseline_gamma as f64).powi(2),
                            ),
                        ] {
                            assert_eq!(got.dim(), expected.dim());
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
fn frame_refresh_uses_the_new_gamma_and_descends_from_any_initial_scale_2825() {
    let (x, decoder, config) = coupled_fixture();
    for orientation in [-1.0, 1.0] {
        let decoder = decoder.mapv(|value| orientation * value);
        for gamma in [0.2, 1.0, 2.0] {
            let mut state =
                BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
            state.gamma = gamma;
            for row in x.outer_iter() {
                state
                    .partial_fit(row.insert_axis(ndarray::Axis(0)))
                    .unwrap();
            }
            // The frame step searches along directions a pass has measured, so the
            // first epoch measures the gradient and leaves the frames in place.
            assert!(!state.end_epoch().unwrap().converged);
            assert_eq!(state.decoder, decoder);
            // The step itself forms on the next pass, started from the same initial
            // scale again.
            state.gamma = gamma;
            for row in x.outer_iter() {
                state
                    .partial_fit(row.insert_axis(ndarray::Axis(0)))
                    .unwrap();
            }
            let stats = state.end_epoch().unwrap();
            assert!(!stats.converged);
            // #2825: the epoch's frames are NO LONGER invariant to the scale it starts
            // from, and that is the corrected behaviour rather than a regression. The
            // support step admits a block only when it lowers `‖x − γ Σ P_g x‖²`, an
            // objective that carries γ; the old invariance held because the top-k gate
            // rule ignored γ entirely, which is the same blindness that let the support
            // step raise the objective the frame and γ steps lower. The scale-free
            // alternative was measured and does NOT close the `K ≫ rank` certificates
            // this rule closes, so the invariance is what gives way.
            //
            // What every seed must still do is what this test is named for: refresh the
            // scale and then DESCEND AT THE REFRESHED SCALE, for either gauge
            // orientation. Both are asserted below, per seed.
            // Recompute the tied codes at each candidate, independently of the
            // stream's moments, holding this epoch's support fixed.
            let loss = |candidate: &Array2<f32>| -> f64 {
                x.outer_iter()
                    .map(|row| {
                        let weights: Vec<f64> = candidate
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
fn tied_frame_update_descends_on_the_frozen_code_ascent_witness_2825() {
    let x = array![[0.4_f32, 4.0], [0.3, -3.0], [-0.3, 3.0]];
    let mut decoder = array![[1.0_f32, -10.0], [10.0, 3.0]];
    for mut row in decoder.outer_iter_mut() {
        let norm = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        row.mapv_inplace(|v| (v as f64 / norm) as f32);
    }
    let (_, _, config) = coupled_fixture();
    let mut state = BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
    let x64 = x.mapv(f64::from);
    let loss = |frame: &Array2<f32>, gamma: f32| {
        let d = frame.mapv(f64::from);
        let residual = &x64 - x64.dot(&d.t()).dot(&d) * gamma as f64;
        residual.iter().map(|v| v * v).sum::<f64>()
    };
    // The first pass only measures the search direction.
    state.partial_fit(x.view()).unwrap();
    assert!(!state.end_epoch().unwrap().converged);
    assert_eq!(state.decoder, decoder);
    state.partial_fit(x.view()).unwrap();
    let first = state.end_epoch().unwrap();
    let proposal = state.decoder.clone();
    let before = loss(&decoder, first.gamma);
    let after = loss(&proposal, first.gamma);
    assert!(
        after < before,
        "actual tied RSS ascended: {before} -> {after}"
    );
    state.partial_fit(x.view()).unwrap();
    let second = state.end_epoch().unwrap();
    let profiled_after = loss(&proposal, second.gamma);
    assert!(profiled_after <= after + 1e-12);
    assert!(profiled_after < before);
}

#[test]
fn tied_projector_moments_match_actual_loss_directional_derivatives() {
    // Exact orthonormal axes make horizontal perturbations independent of
    // f32 frame roundoff. The blocks overlap and have rank two.
    let decoder = array![
        [1.0_f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let x = Array2::from_shape_fn((11, 4), |(i, j)| ((i * 7 + j * 3) as f32).sin());
    let x64 = x.mapv(f64::from);
    let mut config = BlockSparseConfig::new(2, 2);
    config.block_topk = 2;
    config.aux_k = 0;
    for gamma in [0.3, 1.4] {
        let mut state = BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
        state.partial_fit(x.view()).unwrap();
        let data_scale = 2.0 * gamma;
        for block in 0..2 {
            let d = decoder.mapv(f64::from);
            let frame = d.slice(ndarray::s![block * 2..(block + 1) * 2, ..]);
            let raw = Array2::from_shape_fn((2, 4), |(i, j)| ((i * 5 + j + 1) as f64).cos());
            let tangent = &raw - raw.dot(&frame.t()).dot(&frame);
            // The frame's own columns; the rest are the search columns.
            let moment = &state.data_cross[block].slice(ndarray::s![.., ..2]) * data_scale
                + &state.coupling[block].slice(ndarray::s![.., ..2]) * (gamma * gamma);
            let analytic = -2.0
                * moment
                    .iter()
                    .zip(tangent.t().iter())
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
            // The streamed moment is the derivative of the loss AT FIXED SUPPORT, so
            // the numerical reference has to hold the same support. `x D' D` sums every
            // block for every row, which was the same thing only while the router took
            // its top `k` unconditionally; a row now declines a block that does not
            // lower its loss (#2825), and admission boundaries are kinks this central
            // difference would otherwise straddle. Take the support ONCE from the
            // unperturbed frames, then re-project it at each candidate.
            // Route at the scale the STREAM routed at, not the scale this loop prices
            // the moments with: the accumulators were built under `state.gamma`, and a
            // reference routed at a different scale would fix a different support.
            let (blocks, gates, _) = crate::sparse_dict::block_sparse_dictionary_transform(
                x.view(),
                decoder.view(),
                state.gamma,
                config.block_size,
                config.block_topk,
                config.block_tile,
            )
            .expect("route the fixed support from the unperturbed frames");
            let loss = |step: f64| {
                let mut candidate = d.clone();
                let perturbed = &frame.to_owned() + &tangent * step;
                candidate
                    .slice_mut(ndarray::s![block * 2..(block + 1) * 2, ..])
                    .assign(&perturbed);
                (0..x64.nrows())
                    .map(|row| {
                        let mut reconstruction = vec![0.0f64; x64.ncols()];
                        for slot in 0..blocks.ncols() {
                            if gates[[row, slot]] == 0.0 {
                                continue;
                            }
                            let selected = blocks[[row, slot]] as usize;
                            for axis in 0..config.block_size {
                                let direction = candidate.row(selected * config.block_size + axis);
                                let weight: f64 = direction
                                    .iter()
                                    .zip(x64.row(row).iter())
                                    .map(|(&u, &value)| u * value)
                                    .sum();
                                for (accumulated, &u) in
                                    reconstruction.iter_mut().zip(direction.iter())
                                {
                                    *accumulated += gamma * weight * u;
                                }
                            }
                        }
                        x64.row(row)
                            .iter()
                            .zip(reconstruction.iter())
                            .map(|(value, fitted)| (value - fitted).powi(2))
                            .sum::<f64>()
                    })
                    .sum::<f64>()
            };
            let step = 1e-5;
            let numerical = (loss(step) - loss(-step)) / (2.0 * step);
            assert!(
                (analytic - numerical).abs() < 1e-7 * (1.0 + analytic.abs()),
                "block {block}, gamma {gamma}: analytic {analytic}, numerical {numerical}"
            );
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
    let mut state = BlockSparseStreamState::new_with_decoder(candidate, &config).unwrap();
    state.pending_birth = Some(super::PendingBlockBirth {
        block: 0,
        baseline_decoder: baseline.clone(),
        baseline_gamma: 1.0,
        baseline_rss: 0.0,
        baseline_rows: 0,
        baseline_usage: vec![0; 2],
        baseline_second: (0..2).map(|_| Array2::zeros((1, 1))).collect(),
        baseline_supports: Vec::new(),
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
fn a_frame_step_without_search_directions_cannot_certify_a_nonstationary_frame() {
    let (x, decoder, config) = coupled_fixture();
    // A state without search directions cannot move its frames while the actual
    // tied objective still has a gradient. The gradient certificate must see
    // through the zero displacement.
    let mut first = BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
    first.partial_fit(x.view()).unwrap();
    let measured = first.end_epoch().unwrap();
    let mut state = BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
    state.gamma = measured.gamma;
    state.prev_ev = measured.explained_variance;
    state.epochs_run = 1;
    state.partial_fit(x.view()).unwrap();
    let stats = state.end_epoch().unwrap();
    let displacement = crate::sparse_dict::block::frame_fixed_point_residual(
        decoder.view(),
        state.decoder.view(),
        2,
        1,
    )
    .unwrap();
    assert!(displacement <= config.tolerance);
    assert!(stats.gamma_residual <= config.tolerance);
    assert!(stats.frame_residual > config.tolerance);
    assert!(!stats.converged);
    assert!(state.finalize().is_err());
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
    // The contract this control exists for is the one its NAME states: an equal EV
    // cannot certify while gamma OR the frames are still moving. Requiring BOTH to be
    // open was incidental to a router that re-derived a different support every epoch;
    // with the support step descending the shared objective the scale settles first, so
    // the conjunction would pin the old churn rather than the contract. Name which
    // residual is holding the certificate open, so a future change that closes BOTH —
    // which WOULD make this fixture vacuous — fails here instead of passing silently.
    let gamma_open = stats.gamma_residual > config.tolerance;
    let frame_open = stats.frame_residual > config.tolerance;
    assert!(
        gamma_open || frame_open,
        "equal EV with both residuals closed is a converged fit, not this control: \
         gamma_residual={} frame_residual={} tol={}",
        stats.gamma_residual,
        stats.frame_residual,
        config.tolerance
    );
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

#[test]
fn rerouted_frame_trials_cannot_decrease_streamed_explained_variance_2825() {
    // The production failure needs more than one over-complete block.  This
    // deterministic small analogue deliberately has overlapping rank-two
    // structure, six frames in R^8, top-k three, and no birth machinery.
    let (rows, p, g, b) = (96usize, 8usize, 6usize, 2usize);
    let mut bits = 0x2825_u64;
    let x = Array2::from_shape_fn((rows, p), |(row, column)| {
        bits = bits.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let noise = ((bits >> 40) as f64 / (1_u64 << 24) as f64 - 0.5) * 0.05;
        let phase = row as f64 * (column + 1) as f64 / rows as f64;
        (phase.sin() + (phase * 2.0 + column as f64).cos() + noise) as f32
    });
    let config = BlockSparseConfig {
        n_blocks: g,
        block_size: b,
        block_topk: 3,
        max_epochs: 40,
        minibatch: rows,
        block_tile: g,
        frame_ridge: 0.0,
        aux_k: 0,
        matryoshka_prefix: false,
        tolerance: 1.0e-8,
    };
    let mut state = BlockSparseStreamState::new(x.slice(ndarray::s![..24, ..]), &config)
        .expect("over-complete stream");
    let mut previous = f64::NEG_INFINITY;
    let mut measurable_improvements = 0usize;
    for epoch in 0..config.max_epochs {
        state.partial_fit(x.view()).expect("stream fixture");
        let stats = state.end_epoch().expect("close epoch");
        assert!(
            stats.explained_variance >= previous,
            "paired rerouting admitted a loss increase at epoch {epoch}: {previous:.17e} -> {:.17e}",
            stats.explained_variance
        );
        if stats.explained_variance > previous && previous.is_finite() {
            measurable_improvements += 1;
        }
        previous = stats.explained_variance;
        if stats.converged {
            break;
        }
    }
    assert!(
        measurable_improvements > 0,
        "monotonicity would be vacuous if the frame lane never improved"
    );
}

#[test]
fn tied_row_moment_kernel_is_bit_identical_to_the_indexed_loop_2826() {
    // `partial_fit`'s tied projector moments moved from a per-element
    // `[[c, axis]]` loop to a two-pass slice kernel for speed only. The moments
    // feed the frame step, its stationarity certificate and the paired frame
    // trials, so the kernel must perform the old loop's floating-point
    // operations in the same order: bit equality, not a tolerance. The search
    // columns ride alongside and must leave the frame columns' operations alone.
    let mut bits = 0x2826_u64;
    let mut draw = move || {
        bits = bits
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (bits >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
    };
    for (b, p, k) in [(1usize, 5usize, 1usize), (2, 37, 8), (3, 64, 3)] {
        let entries = 23usize;
        // Rows are views into a transposed matrix, so each has a non-unit
        // stride, as a Fortran-ordered shard does.
        let columns = Array2::from_shape_fn((p, entries), |_| (4.0 * draw()) as f32);
        let rows = columns.t();
        let decoder = Array2::from_shape_fn((b, p), |_| draw() as f32);
        let codes = Array2::from_shape_fn((entries, b), |_| 3.0 * draw());
        let sums = Array2::from_shape_fn((entries, p), |_| 2.0 * draw());
        let frame: Vec<f64> = decoder.iter().map(|&value| f64::from(value)).collect();
        let searched = Array2::from_shape_fn((2 * b, p), |_| draw() as f32);
        let directions: Vec<f64> = searched.iter().map(|&value| f64::from(value)).collect();

        let mut coupling = vec![0.0_f64; p * b];
        let mut data_cross = vec![0.0_f64; p * b];
        let mut searched_coupling = vec![0.0_f64; p * 3 * b];
        let mut searched_data = vec![0.0_f64; p * 3 * b];
        let mut v = vec![0.0_f64; p];
        let mut coordinates = vec![0.0_f64; 2 * b];
        let mut searched_coordinates = vec![0.0_f64; 6 * b];
        let mut reference_coupling = Array2::<f64>::zeros((p, b));
        let mut reference_data = Array2::<f64>::zeros((p, b));
        let mut merged_coupling = Array2::<f64>::zeros((p, b));
        for entry in 0..entries {
            let w = codes.row(entry).to_vec();
            let sum = sums.row(entry).to_vec();
            let xi = rows.row(entry);
            assert_ne!(xi.strides()[0], 1, "the fixture must exercise a strided row");
            super::accumulate_tied_row_moments(
                &frame,
                &[],
                &w,
                xi,
                &sum,
                k as f64,
                &mut v,
                &mut coordinates,
                &mut coupling,
                &mut data_cross,
            );
            super::accumulate_tied_row_moments(
                &frame,
                &directions,
                &w,
                xi,
                &sum,
                k as f64,
                &mut v,
                &mut searched_coordinates,
                &mut searched_coupling,
                &mut searched_data,
            );

            // The indexed loop, with `decoder[[block * b + axis, c]]` narrowed to this
            // one block's rows. `w` decodes the block's projection; the moments take
            // the stored rows' inner products `Ux` (#2502), summed in `c` order.
            let mut inner = vec![0.0; b];
            for c in 0..p {
                for (axis, value) in inner.iter_mut().enumerate() {
                    *value += decoder[[axis, c]] as f64 * xi[c] as f64;
                }
            }
            let mut reference_v_coordinates = vec![0.0; b];
            for c in 0..p {
                let mut own = 0.0;
                for (axis, &weight) in w.iter().enumerate() {
                    own += weight * decoder[[axis, c]] as f64;
                }
                let value = k as f64 * own - sum[c];
                let x = xi[c] as f64;
                for (axis, &product) in inner.iter().enumerate() {
                    reference_coupling[[c, axis]] += value * product;
                    reference_data[[c, axis]] += x * product;
                    reference_v_coordinates[axis] += decoder[[axis, c]] as f64 * value;
                }
            }
            for c in 0..p {
                for axis in 0..b {
                    reference_coupling[[c, axis]] += xi[c] as f64 * reference_v_coordinates[axis];
                }
            }
            for c in 0..p {
                for axis in 0..b {
                    reference_coupling[[c, axis]] -= k as f64 * xi[c] as f64 * inner[axis];
                }
            }
            // Positive control: the same two coupling terms added as ONE sum per
            // element. A fixture that cannot tell this association from the
            // original order would make the bit equality below vacuous.
            for c in 0..p {
                let mut own = 0.0;
                for (axis, &weight) in w.iter().enumerate() {
                    own += weight * decoder[[axis, c]] as f64;
                }
                let value = k as f64 * own - sum[c];
                for (axis, &product) in inner.iter().enumerate() {
                    merged_coupling[[c, axis]] +=
                        value * product + xi[c] as f64 * reference_v_coordinates[axis];
                }
            }
            for c in 0..p {
                for axis in 0..b {
                    merged_coupling[[c, axis]] -= k as f64 * xi[c] as f64 * inner[axis];
                }
            }
        }
        for c in 0..p {
            for axis in 0..b {
                for (got, expected, name) in [
                    (coupling[c * b + axis], reference_coupling[[c, axis]], "coupling"),
                    (data_cross[c * b + axis], reference_data[[c, axis]], "data_cross"),
                    (
                        searched_coupling[c * 3 * b + axis],
                        reference_coupling[[c, axis]],
                        "searched coupling",
                    ),
                    (
                        searched_data[c * 3 * b + axis],
                        reference_data[[c, axis]],
                        "searched data_cross",
                    ),
                ] {
                    assert_eq!(
                        got.to_bits(),
                        expected.to_bits(),
                        "{name}[{c}, {axis}] differs (b={b}, p={p}, k={k}): {got:e} vs {expected:e}"
                    );
                }
            }
        }
        assert!(
            merged_coupling
                .iter()
                .zip(reference_coupling.iter())
                .any(|(merged, original)| merged.to_bits() != original.to_bits()),
            "b={b}, p={p}, k={k}: the fixture does not resolve an association change, so \
             bit equality with the original loop would not distinguish the two orders"
        );
    }
}

#[test]
fn a_rejected_frame_trial_stashes_gamma_scaled_moments_for_rank_charges() {
    // Two overlapping unit frames in R^2 and rows x = (1, 1). Admission is descent in
    // the tied loss at the stored gamma: [0.6, 0.8] enters first (gate 1.4), and at
    // gamma = 0.5 the frame [1, 0] lowers the loss too (-(2γ-γ²)·1 + 2γ²·0.84 < 0). Both
    // blocks are admitted, their projections overlap, and the profiled gamma (about
    // 0.64) is well away from one. The proposal swaps the two blocks: it represents the
    // same model, its RSS equals the baseline's exactly, and the equality rule rejects it.
    let x = Array2::from_elem((17, 2), 1.0_f32);
    let baseline = array![[1.0_f32, 0.0], [0.6, 0.8]];
    let proposal = array![[0.6_f32, 0.8], [1.0, 0.0]];
    let mut config = BlockSparseConfig::new(2, 1);
    config.block_topk = 2;
    config.minibatch = 5;
    config.aux_k = 0;
    let mut state = BlockSparseStreamState::new_with_decoder(proposal.clone(), &config).unwrap();
    state.gamma = 0.5;
    state.pending_frame = Some(super::PendingFrameTrial {
        baseline_decoder: baseline,
        baseline_gamma: 0.5,
        proposed_decoder: proposal,
        baseline_rss: 0.0,
        baseline_gamma_num: 0.0,
        baseline_gamma_den: 0.0,
        baseline_rows: 0,
        baseline_usage: vec![0; 2],
        baseline_second: vec![Array2::zeros((1, 1)); 2],
        baseline_supports: Vec::new(),
        rerouted_rows: 0,
        moves: super::TrialMoves::default(),
    });
    state.partial_fit(x.view()).unwrap();
    let pending = state.pending_frame.as_ref().unwrap();
    let gamma = (pending.baseline_gamma_num / pending.baseline_gamma_den) as f32 as f64;
    assert!(
        (gamma * gamma - 1.0).abs() > 0.1,
        "gamma^2 = {} would not tell scaled from unscaled moments",
        gamma * gamma
    );
    let expected: Vec<f64> = pending
        .baseline_second
        .iter()
        .map(|second| second[[0, 0]] * (gamma * gamma))
        .collect();
    assert!(expected.iter().all(|&value| value > 0.0));
    let stats = state.end_epoch().unwrap();
    assert!(
        stats.frame_residual.is_infinite(),
        "the swapped proposal must be rejected, not stepped"
    );
    assert_eq!(state.gamma as f64, gamma);
    for (block, &target) in expected.iter().enumerate() {
        let stashed = state.last_second[block][[0, 0]];
        assert!(
            (stashed - target).abs() <= 1e-12 * target,
            "block {block}: stashed {stashed:e}, expected the gamma^2-scaled {target:e}"
        );
    }
}

#[test]
fn a_row_keeps_its_retained_support_unless_the_routed_support_lowers_its_loss_2502() {
    // Blocks 0 and 1 are the same line and block 2 sits at 60 degrees to it. The
    // routed support of x = e0 is one of the duplicates. Keeping the other duplicate
    // costs nothing a rounding band resolves, so the row keeps it. Keeping block 2
    // leaves 3/4 of the row's energy unexplained, so the row adopts the routed support.
    let x = array![[1.0_f32, 0.0]];
    let decoder = array![[1.0_f32, 0.0], [1.0, 0.0], [0.5, 3.0_f32.sqrt() / 2.0]];
    let route = || {
        crate::sparse_dict::block::route_and_code_all(x.view(), decoder.view(), 1.0, 3, 1, 1, 1, 3)
            .expect("route the row")
            .remove(0)
    };
    let routed = route().blocks[0];
    assert!(routed < 2, "the router must pick a duplicate, got block {routed}");
    let duplicate = 1 - routed;
    let spans = crate::sparse_dict::block::stored_spans(decoder.view(), 1)
        .expect("every block spans one dimension");
    let (kept, kept_projection, declined) = super::descend_block_support(
        x.row(0),
        decoder.view(),
        &spans,
        1.0,
        1,
        1,
        route(),
        &[duplicate],
    );
    assert_eq!(kept.blocks, vec![duplicate]);
    assert_ne!(kept.gates[0], 0.0);
    assert_eq!(kept_projection.rss, 0.0);
    // The routed duplicate admits a different block and resolves no loss decrease, so
    // it is declined at zero excess.
    assert_eq!(declined, Some(0.0));
    let (adopted, adopted_projection, adopted_declined) = super::descend_block_support(
        x.row(0),
        decoder.view(),
        &spans,
        1.0,
        1,
        1,
        route(),
        &[2],
    );
    assert_eq!(adopted.blocks, vec![routed]);
    assert_eq!(adopted_projection.rss, 0.0);
    assert_eq!(adopted_declined, None);
    // A row no committed pass has seen takes its routed support.
    let (first, _, first_declined) = super::descend_block_support(
        x.row(0),
        decoder.view(),
        &spans,
        1.0,
        1,
        1,
        route(),
        &[super::NO_BLOCK],
    );
    assert_eq!(first.blocks, vec![routed]);
    assert_eq!(first_declined, None);
}

#[test]
fn a_pass_that_streams_different_rows_is_refused_2502() {
    let (x, decoder, config) = coupled_fixture();
    let mut state = BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
    state.partial_fit(x.view()).unwrap();
    state.end_epoch().unwrap();
    let mut reordered = x.clone();
    reordered.invert_axis(ndarray::Axis(0));
    let error = state.partial_fit(reordered.view()).unwrap_err();
    assert!(error.contains("same rows in the same order"), "{error}");

    let mut shortened = BlockSparseStreamState::new_with_decoder(decoder, &config).unwrap();
    shortened.partial_fit(x.view()).unwrap();
    shortened.end_epoch().unwrap();
    shortened.partial_fit(x.slice(ndarray::s![..2, ..])).unwrap();
    let error = shortened.end_epoch().unwrap_err();
    assert!(error.contains("same rows in the same order"), "{error}");
}

/// A deterministic generator of `f32` block frames, each Gram–Schmidt orthonormalised
/// and stored as `f32`, as the stream stores them.
fn stored_frames(blocks: usize, b: usize, p: usize, seed: u64) -> Array2<f32> {
    let mut bits = seed;
    let mut decoder = Array2::<f32>::zeros((blocks * b, p));
    for block in 0..blocks {
        let mut frame = Array2::from_shape_fn((b, p), |_| {
            bits = bits
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((bits >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) as f32
        });
        crate::sparse_dict::block::gram_schmidt_rows(&mut frame);
        decoder
            .slice_mut(ndarray::s![block * b..(block + 1) * b, ..])
            .assign(&frame);
    }
    decoder
}

#[test]
fn a_retained_support_prices_its_stored_span_2502() {
    // Doubling one stored row is exact in binary floating point and leaves the block's
    // span bit for bit. A retained support must then price every row to the same loss
    // and gate, to the bit: the stream's trials priced through `UᵀU` moved with every
    // f32 rewrite of a frame instead (#2502, lane job 1229606). The positive control is
    // the `‖Ux‖` gate that pricing used, which the doubling moves.
    let (b, p, blocks, k) = (2usize, 6usize, 4usize, 3usize);
    let decoder = stored_frames(blocks, b, p, 0x2502);
    let mut doubled = decoder.clone();
    for column in 0..p {
        doubled[[b, column]] *= 2.0; // block 1's first row
    }
    let rows = stored_frames(8, 2, p, 0x2503);
    let prior = [1_u32, 0, 3];
    let gamma = 0.8_f32;
    let price = |frames: &Array2<f32>, row: ndarray::ArrayView1<'_, f32>| {
        let spans = crate::sparse_dict::block::stored_spans(frames.view(), b)
            .expect("every block spans two dimensions");
        let code =
            super::kept_block_code(row, frames.view(), &spans.inverse_grams, gamma, b, k, &prior);
        let rss = super::row_projection(row, frames.view(), &code, b, gamma).rss;
        (code, rss)
    };
    let mut gram_free_moved = 0usize;
    for row in rows.outer_iter() {
        let (stored, stored_rss) = price(&decoder, row);
        let (rescaled, rescaled_rss) = price(&doubled, row);
        assert_eq!(
            stored_rss.to_bits(),
            rescaled_rss.to_bits(),
            "doubling a stored row moved a retained support's loss: {stored_rss:e} vs {rescaled_rss:e}"
        );
        for slot in 0..prior.len() {
            assert_eq!(stored.gates[slot].to_bits(), rescaled.gates[slot].to_bits());
        }
        let gram_free_gate = |frames: &Array2<f32>| {
            (0..b)
                .map(|axis| {
                    let s: f64 = row
                        .iter()
                        .zip(frames.row(b + axis).iter())
                        .map(|(&x, &a)| x as f64 * a as f64)
                        .sum();
                    s * s
                })
                .sum::<f64>()
                .sqrt()
        };
        if gram_free_gate(&decoder).to_bits() != gram_free_gate(&doubled).to_bits() {
            gram_free_moved += 1;
        }
    }
    assert_eq!(gram_free_moved, rows.nrows(), "the ‖Ux‖ gate must see the doubling on every row");
}

#[test]
fn backtracking_ends_at_the_baseline_bits_2502() {
    // Re-orthonormalising an f32-stored frame need not return its bits. The old
    // bisection re-orthonormalised every block of every midpoint, so on the Spark
    // layer-18 fit all 274 trials from epoch 2727 to 3000 were the same rewrite of all
    // 1024 blocks and backtracking never returned to the baseline (#2502, run 1165180).
    // Backtracking must leave unmoved blocks' bits alone and end at the baseline.
    let (b, p, blocks) = (2usize, 16usize, 32usize);
    let baseline = stored_frames(blocks, b, p, 0x2727);
    let block_rows = |frames: &Array2<f32>, block: usize| {
        frames
            .slice(ndarray::s![block * b..(block + 1) * b, ..])
            .to_owned()
    };
    // Positive control: re-rounding under Gram–Schmidt moves some stored block's bits,
    // which is what kept the old bisection from ever reaching the baseline.
    let rerounded = (0..blocks)
        .filter(|&block| {
            let stored = block_rows(&baseline, block);
            let mut again = stored.clone();
            crate::sparse_dict::block::gram_schmidt_rows(&mut again);
            again != stored
        })
        .count();
    assert!(
        rerounded > 0,
        "no fixture block re-rounds under Gram–Schmidt, so the fixture cannot see the old cycle"
    );
    // The proposal moves block 0 by a resolvable step and keeps every other block's bits.
    let mut proposal = baseline.clone();
    let mut moved = block_rows(&stored_frames(1, b, p, 0x3000), 0) * 0.3 + block_rows(&baseline, 0);
    crate::sparse_dict::block::gram_schmidt_rows(&mut moved);
    proposal
        .slice_mut(ndarray::s![..b, ..])
        .assign(&moved);
    let mut trial = proposal;
    let mut halvings = 0usize;
    while trial != baseline {
        let midpoint = super::bisect_frame_trial(&baseline, &trial, b).expect("bisect the trial");
        for block in 1..blocks {
            assert_eq!(
                block_rows(&midpoint, block),
                block_rows(&baseline, block),
                "unmoved block {block} changed its bits at halving {halvings}"
            );
        }
        trial = midpoint;
        halvings += 1;
        // A float32 frame halves its displacement at most about 24 times before storage
        // stops resolving it; far past that, backtracking is cycling.
        assert!(halvings <= 64, "backtracking did not reach the baseline bits in 64 halvings");
    }
    assert!(halvings > 1, "block 0's move must take several halvings to fall below resolution");
}

#[test]
fn a_ritz_step_below_storage_resolution_keeps_the_stored_bits_2502() {
    // H = diag(2, 1, −g, −g) + ε(e1e2ᵀ + e2e1ᵀ) on span[U, R], U = [e0, e1], R = [e2, e3].
    // The top Ritz pair rotates e1 toward e2 by θ ≈ ε/(1 + g) and raises the surrogate
    // trace by εθ ≈ θ²g. At θ = 1.2e-7 the stored projector moves by about 0.71θ, below
    // f32 resolution, while f64 resolves the raise about 65 times over its eigensolver
    // error ε₆₄·g. So the step takes the raise branch and must keep U's stored bits
    // (#2502). At θ = 1.2e-5 the move is resolvable and the frame must move.
    let (b, p) = (2usize, 4usize);
    let g = 1.0e6_f64;
    let step = |theta: f64| {
        let coupling = theta * (1.0 + g);
        let current = array![[1.0_f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let directions = array![
            [0.0_f32, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        let mut h = Array2::<f64>::zeros((p, p));
        h[[0, 0]] = 2.0;
        h[[1, 1]] = 1.0;
        h[[2, 2]] = -g;
        h[[3, 3]] = -g;
        h[[1, 2]] = coupling;
        h[[2, 1]] = coupling;
        // H·[U, R, P]: every live basis vector is a coordinate axis, so its image is a
        // column of H; the two zero search rows have zero images.
        let mut action = Array2::<f64>::zeros((p, 3 * b));
        for column in 0..p {
            action.column_mut(column).assign(&h.column(column));
        }
        let mut proposal = Array2::<f32>::zeros((b, p));
        let mut next = Array2::<f32>::zeros((2 * b, p));
        crate::sparse_dict::block_frame::ritz_tied_frame_step(
            current.view(),
            directions.view(),
            action.view(),
            Array2::<f64>::zeros((b, b)).view(),
            1.0,
            0.0,
            proposal.view_mut(),
            next.view_mut(),
        )
        .expect("the Ritz step on a resolved subspace");
        (current, proposal)
    };
    let (current, kept) = step(1.2e-7);
    assert_eq!(kept, current, "a sub-resolution Ritz step rewrote the stored frame");
    let (current, stepped) = step(1.2e-5);
    assert_ne!(stepped, current, "a resolvable Ritz step must move the frame");
}

#[test]
fn a_frame_trial_commits_on_its_paired_decrease_over_the_rows_it_changes_2502() {
    // Ten rows lie on the line (1, 1e-4, 0, 0) and ten thousand lie in the e2-e3 plane
    // with a large e3 component no block spans. Block 0, a line in the e0-e1 plane, serves
    // the ten; block 1 = e2 serves the ten thousand; no row admits both. The proposal rotates
    // block 0 halfway toward the ten rows' line and lowers their loss by about 7.5e-8, inside
    // the corpus-wide band of about 4.4e-6 that the others' residuals (RSS about 1e8) set.
    // Those rows admit only the unmoved block 1 in both passes and are priced to the same
    // bits, so the trial is decided over the ten rows, where the decrease clears its band by
    // orders of magnitude. The corpus-wide rule refused such steps and cycled the Spark fit
    // at a frame residual of 1.163e-4 (#2502, lane job 1248199).
    let mut x = Array2::<f32>::zeros((10_010, 4));
    for row in 0..10 {
        x[[row, 0]] = 1.0;
        x[[row, 1]] = 1.0e-4;
    }
    for row in 10..x.nrows() {
        x[[row, 2]] = 10.0;
        x[[row, 3]] = 100.0;
    }
    let theta = 5.0e-5_f64;
    let baseline = array![[1.0_f32, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    let proposal = array![
        [theta.cos() as f32, theta.sin() as f32, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ];
    let mut config = BlockSparseConfig::new(2, 1);
    config.block_topk = 2;
    config.minibatch = 4096;
    config.aux_k = 0;
    let mut state = BlockSparseStreamState::new_with_decoder(proposal.clone(), &config).unwrap();
    state.gamma = 1.0;
    state.pending_frame = Some(super::PendingFrameTrial {
        baseline_decoder: baseline,
        baseline_gamma: 1.0,
        proposed_decoder: proposal,
        baseline_rss: 0.0,
        baseline_gamma_num: 0.0,
        baseline_gamma_den: 0.0,
        baseline_rows: 0,
        baseline_usage: vec![0; 2],
        baseline_second: vec![Array2::zeros((1, 1)); 2],
        baseline_supports: Vec::new(),
        rerouted_rows: 0,
        moves: super::TrialMoves::default(),
    });
    state.partial_fit(x.view()).unwrap();
    let trial = state
        .end_epoch()
        .unwrap()
        .frame_trial
        .expect("the pending trial is adjudicated");
    // Positive control: over the whole corpus the decrease lies inside its band, so a
    // corpus-wide rule refuses the step.
    assert!(
        trial.decrease <= trial.resolution,
        "the corpus decrease {:e} clears its band {:e}, so the fixture cannot tell the rules apart",
        trial.decrease,
        trial.resolution
    );
    assert_eq!(trial.moved_rows, 10);
    assert!(
        trial.moved_decrease > 1.0e6 * trial.moved_resolution,
        "paired decrease {:e} against its band {:e}",
        trial.moved_decrease,
        trial.moved_resolution
    );
    assert!(trial.committed, "a paired decrease far above its band must commit");
}

#[test]
fn a_resumed_stream_continues_bit_for_bit_2502() {
    // A long fit runs as a chain of bounded jobs, each resuming the last one's checkpoint, so a
    // resumed stream must continue exactly as the uninterrupted one does. Checkpointing and
    // resuming at EVERY epoch boundary must reproduce every epoch of the straight run bit for
    // bit. Positive controls: some boundary carries a staged birth and some carries a pending
    // frame trial, the two transactions a checkpoint must carry across.
    let mut seed = 2502_u64;
    let mut sample = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as f32 / 2147483648.0 - 0.5
    };
    let x = Array2::from_shape_fn((37, 8), |_| sample());
    let mut decoder = Array2::from_shape_fn((12, 8), |_| sample());
    for block in 0..5 {
        let mut frame = decoder
            .slice(ndarray::s![block * 2..(block + 1) * 2, ..])
            .to_owned();
        crate::sparse_dict::block::gram_schmidt_rows(&mut frame);
        decoder
            .slice_mut(ndarray::s![block * 2..(block + 1) * 2, ..])
            .assign(&frame);
    }
    // A dead block: its zero frame has the zero projector, so no row admits it and a birth
    // can be staged there.
    decoder.slice_mut(ndarray::s![10.., ..]).fill(0.0);
    let mut config = BlockSparseConfig::new(6, 2);
    config.block_topk = 3;
    config.minibatch = 13;
    config.aux_k = 1;
    let epochs = 8;
    let record = |state: &BlockSparseStreamState, stats: &super::BlockEpochStats| {
        (
            state.decoder.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
            state.gamma.to_bits(),
            stats.explained_variance.to_bits(),
            stats.frame_residual.to_bits(),
            stats.gamma_residual.to_bits(),
            stats.accepted_births,
            stats.converged,
            state.retained_supports.clone(),
        )
    };
    let mut straight = BlockSparseStreamState::new_with_decoder(decoder.clone(), &config).unwrap();
    let mut expected = Vec::new();
    for _ in 0..epochs {
        straight.partial_fit(x.view()).unwrap();
        let stats = straight.end_epoch().unwrap();
        expected.push(record(&straight, &stats));
    }
    let path = std::env::temp_dir().join(format!("gam-block-stream-checkpoint-{}.bin", std::process::id()));
    let mut chained = BlockSparseStreamState::new_with_decoder(decoder, &config).unwrap();
    let mut births = 0usize;
    let mut trials = 0usize;
    for epoch in 0..epochs {
        chained.partial_fit(x.view()).unwrap();
        if epoch == 0 {
            let error = chained.checkpoint(&path).unwrap_err();
            assert!(error.contains("unfinished pass"), "{error}");
        }
        let stats = chained.end_epoch().unwrap();
        assert_eq!(record(&chained, &stats), expected[epoch], "epoch {epoch} diverged after a resume");
        births += usize::from(chained.pending_birth.is_some());
        trials += usize::from(chained.pending_frame.is_some());
        chained.checkpoint(&path).unwrap();
        chained = BlockSparseStreamState::resume(&path, &config).unwrap();
    }
    assert!(births > 0, "no boundary carried a staged birth, so the checkpoint never had to");
    assert!(trials > 0, "no boundary carried a pending frame trial, so the checkpoint never had to");
    let mut other = config;
    other.tolerance *= 2.0;
    let error = BlockSparseStreamState::resume(&path, &other)
        .err()
        .expect("a checkpoint written under another tolerance must be refused");
    assert!(error.contains("tolerance"), "{error}");
    // The epoch cap is the driver's, so a chain may resume under another one.
    let mut extended = config;
    extended.max_epochs += 7;
    assert!(
        BlockSparseStreamState::resume(&path, &extended).is_ok(),
        "a stream driven to another epoch cap must resume"
    );
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn a_block_that_meets_the_certificate_bar_keeps_its_stored_bits_2502() {
    // H = diag(2, 1, −g) + ε(e1e2ᵀ + e2e1ᵀ) on span[U, R], U = [e0, e1], R = [e2]: the top
    // Ritz pair rotates e1 toward e2 by a resolvable angle, and the stationarity certificate
    // is ε over the action's norm. A bar at or above that certificate means the block already
    // passes it, so its step can only shift its neighbors' conditional optima: it must keep
    // U's stored bits, with its tangent as the next search row. A bar just below it must let
    // the block move (#2502).
    let (b, p) = (2usize, 3usize);
    let g = 10.0_f64;
    let coupling = 0.05_f64;
    let current = array![[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let directions = array![[0.0_f32, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let mut h = Array2::<f64>::zeros((p, p));
    h[[0, 0]] = 2.0;
    h[[1, 1]] = 1.0;
    h[[2, 2]] = -g;
    h[[1, 2]] = coupling;
    h[[2, 1]] = coupling;
    let mut action = Array2::<f64>::zeros((p, 3 * b));
    for column in 0..p {
        action.column_mut(column).assign(&h.column(column));
    }
    let step = |bar: f64| {
        let mut proposal = Array2::<f32>::zeros((b, p));
        let mut next = Array2::<f32>::zeros((2 * b, p));
        let stationarity = crate::sparse_dict::block_frame::ritz_tied_frame_step(
            current.view(),
            directions.view(),
            action.view(),
            Array2::<f64>::zeros((b, b)).view(),
            1.0,
            bar,
            proposal.view_mut(),
            next.view_mut(),
        )
        .expect("the Ritz step on a resolved subspace");
        (stationarity, proposal, next)
    };
    let (stationarity, moved, _) = step(0.0);
    assert!(stationarity > 0.0);
    assert_ne!(moved, current, "with no bar the resolvable step must move the frame");
    let (_, kept, next) = step(stationarity);
    assert_eq!(kept, current, "a block that meets the bar must keep its stored bits");
    assert_eq!(
        next.row(1),
        array![0.0_f32, 0.0, 1.0].view(),
        "a kept block's next search row is its unit tangent gradient"
    );
    let (_, below, _) = step(0.5 * stationarity);
    assert_ne!(below, current, "a block above the bar must still move");
}

#[test]
fn a_rescaled_frame_trial_is_priced_by_its_span_and_descends_2502() {
    // Block 0 is a line that starts on e0; block 1 is e2; every row admits both. The
    // proposal rotates block 0 halfway toward the rows' principal axis in the e0-e1 plane,
    // a real descent. Stored at norm 1 or at norm 2 (doubling is exact, so the span is the
    // same bits), it must commit with the same paired decrease: the projector
    // Uᵀ(UUᵀ)⁻¹U is a function of the span. Priced as UᵀU instead, the norm-2 frame
    // reconstructs every row four times too far along the line and the trial ascends
    // (#2502).
    let rows = 16usize;
    let x = Array2::from_shape_fn((rows, 3), |(row, column)| {
        let i = row as f64;
        let angle = std::f64::consts::PI / 6.0 + 0.2 * i.sin();
        let radius = 1.0 + 0.3 * (3.0 * i).cos();
        (match column {
            0 => radius * angle.cos(),
            1 => radius * angle.sin(),
            _ => 0.5 + 0.2 * (2.0 * i).sin(),
        }) as f32
    });
    let baseline = array![[1.0_f32, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let theta = std::f64::consts::PI / 12.0;
    let mut config = BlockSparseConfig::new(2, 1);
    config.block_topk = 2;
    config.minibatch = rows;
    config.aux_k = 0;
    let trial = |norm: f32| {
        let proposal = array![
            [norm * theta.cos() as f32, norm * theta.sin() as f32, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let mut state = BlockSparseStreamState::new_with_decoder(proposal.clone(), &config).unwrap();
        state.gamma = 1.0;
        state.pending_frame = Some(super::PendingFrameTrial {
            baseline_decoder: baseline.clone(),
            baseline_gamma: 1.0,
            proposed_decoder: proposal,
            baseline_rss: 0.0,
            baseline_gamma_num: 0.0,
            baseline_gamma_den: 0.0,
            baseline_rows: 0,
            baseline_usage: vec![0; 2],
            baseline_second: vec![Array2::zeros((1, 1)); 2],
            baseline_supports: Vec::new(),
            rerouted_rows: 0,
            moves: super::TrialMoves::default(),
        });
        state.partial_fit(x.view()).unwrap();
        assert_eq!(state.usage, vec![rows, rows], "every row must admit both blocks");
        state
            .end_epoch()
            .unwrap()
            .frame_trial
            .expect("the pending trial is adjudicated")
    };
    let unit = trial(1.0);
    let doubled = trial(2.0);
    assert!(unit.committed && unit.moved_decrease > 0.0, "{unit:?}");
    assert!(doubled.committed, "the norm-2 frame of a descending step must commit: {doubled:?}");
    assert!(
        (doubled.moved_decrease - unit.moved_decrease).abs() <= 1.0e-12 * unit.moved_decrease,
        "the stored norm moved the priced decrease: {:e} vs {:e}",
        doubled.moved_decrease,
        unit.moved_decrease
    );
}
