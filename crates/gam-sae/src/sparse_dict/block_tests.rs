//! Tests for the block-sparse lane. The load-bearing one is
//! [`gauge_invariant_selection_and_loss_under_block_rotation`]: rotating a
//! block's internal basis by a random `O(b)` matrix must leave every gate,
//! the block selection, and the loss unchanged — the invariance the whole
//! design rests on.

use super::*;
use crate::frames::GrassmannFrame;
use crate::sparse_dict::{
    BlockChartComposeConfig, BlockSeedManifestConfig, block_sparse_dictionary_firings,
    block_sparse_dictionary_seed_manifest, compose_block_coordinate_charts,
};
use ndarray::{Array1, Array2};

/// Exact tied loss of a stored row code, independent of the fitter's own
/// accumulation, so the assertions below price the objective and not a proxy.
fn row_loss(
    row: ArrayView1<'_, f32>,
    code: &RowBlockCode,
    decoder: ArrayView2<'_, f32>,
    b: usize,
) -> f64 {
    let reconstruction = reconstruct_stored_code_row(code, decoder, b);
    row.iter()
        .zip(reconstruction.iter())
        .map(|(value, fitted)| {
            let residual = *value as f64 - *fitted as f64;
            residual * residual
        })
        .sum()
}

/// #2825 — A ROW TAKES A BLOCK ONLY WHEN THAT BLOCK LOWERS ITS LOSS.
///
/// `k` is a cap, not a quota. Two blocks 15 degrees apart both carry a large
/// gate against `x = (1, 0)`, so the gate ranking admits both; the second is
/// almost the first, and at `γ = 1` admitting it moves the reconstruction from
/// exactly `x` to `(1.933, 0.25)` — the loss rises from `0` to `0.933`. The
/// separable ranking cannot see that, because the term it omits,
/// `γ² Σ_{g≠h} y_g·y_h`, is the entire disagreement. This is the mechanism
/// behind the 91% re-routing giveback measured in a00be6eb0.
#[test]
fn a_block_is_admitted_only_when_it_lowers_the_row_loss_2825() {
    let theta = std::f64::consts::PI / 12.0; // 15 degrees
    let decoder = ndarray::array![[1.0_f32, 0.0], [theta.cos() as f32, theta.sin() as f32]];
    let x = ndarray::array![1.0_f32, 0.0];
    let gate_one = (x[0] as f64 * decoder[[1, 0]] as f64).abs();
    // NON-VACUITY: the overlapping block is a genuine top-k candidate.
    assert!(
        gate_one > 0.9,
        "the overlapping block must rank, or nothing is being refused"
    );
    let shortlist = [(0u32, 1.0_f32), (1u32, gate_one as f32)];

    let selected = code_row(x.view(), decoder.view(), 1.0, 1, 2, &shortlist);
    assert_eq!(selected.blocks[0], 0);
    assert_eq!(
        selected.gates[1], 0.0,
        "the second slot must be canonical padding, not a firing"
    );
    assert_eq!(selected.codes[1], 0.0);
    assert_eq!(selected.projections[1], 0.0);
    assert!(row_loss(x.view(), &selected, decoder.view(), 1) <= 1.0e-12);

    // The unconditional top-k support, priced the same way, is strictly worse.
    let forced = RowBlockCode {
        blocks: vec![0, 1],
        gates: vec![1.0, gate_one as f32],
        codes: vec![1.0, gate_one as f32],
        projections: vec![1.0, gate_one],
    };
    let forced_loss = row_loss(x.view(), &forced, decoder.view(), 1);
    assert!(
        forced_loss > 0.9,
        "the refused support must be the strictly worse one; got {forced_loss}"
    );
}

/// The refusal above is a property of OVERLAP, not a blanket cap: orthogonal
/// blocks that each lower the loss are all admitted, and together they
/// reconstruct the row exactly.
#[test]
fn orthogonal_blocks_that_lower_the_loss_are_all_admitted_2825() {
    let decoder = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
    let x = ndarray::array![3.0_f32, 4.0];
    let shortlist = [(1u32, 4.0_f32), (0u32, 3.0_f32)];
    let selected = code_row(x.view(), decoder.view(), 1.0, 1, 2, &shortlist);
    let mut taken: Vec<u32> = selected
        .blocks
        .iter()
        .zip(selected.gates.iter())
        .filter_map(|(block, gate)| (*gate != 0.0).then_some(*block))
        .collect();
    taken.sort_unstable();
    assert_eq!(
        taken,
        vec![0, 1],
        "an orthogonal pair must still take both blocks"
    );
    assert!(row_loss(x.view(), &selected, decoder.view(), 1) <= 1.0e-10);
}

/// #2825 — THE NULL OF THIS MODEL IS `γ = 0`, NOT AN EMPTY SUPPORT.
///
/// For `γ ≥ 2` the admission weight `2γ−γ²` is non-positive and NO block lowers
/// the loss, so an unguarded descent rule would empty every row — after which
/// `refresh_gamma` divides by a zero denominator and the fit cannot come back.
/// The first admission is therefore unconditional: the row keeps the single
/// block that best explains it, the scale stays definable, and `γ = 0` remains
/// the way to say "explain nothing". Below `γ = 2` the guard is inert, which is
/// every scale the fit actually visits.
#[test]
fn a_non_descent_scale_keeps_the_support_definable_2825() {
    let decoder = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
    let x = ndarray::array![3.0_f32, 4.0];
    let shortlist = [(1u32, 4.0_f32), (0u32, 3.0_f32)];
    let live = |code: &RowBlockCode| -> Vec<u32> {
        code.blocks
            .iter()
            .zip(code.gates.iter())
            .filter_map(|(block, gate)| (*gate != 0.0).then_some(*block))
            .collect()
    };
    // NON-VACUITY: at a descent scale this row takes both blocks.
    assert_eq!(
        live(&code_row(x.view(), decoder.view(), 1.0, 1, 2, &shortlist)).len(),
        2
    );
    for gamma in [2.0_f32, 3.5] {
        assert!(
            2.0 * gamma - gamma * gamma <= 0.0,
            "gamma {gamma} must be non-descent"
        );
        let selected = code_row(x.view(), decoder.view(), gamma, 1, 2, &shortlist);
        assert_eq!(
            live(&selected),
            vec![1],
            "a non-descent scale must keep exactly the strongest block, at gamma {gamma}"
        );
        // A definable scale: the gamma-free projection sum is non-zero, which is
        // the denominator `refresh_gamma` divides by.
        let projected: f64 = selected.projections.iter().map(|w| w * w).sum();
        assert!(projected > 0.0, "the refreshed scale must stay defined");
    }
    // The null is reached through the scale, and it decodes to exactly zero.
    let null = code_row(x.view(), decoder.view(), 0.0, 1, 2, &shortlist);
    assert_eq!(live(&null), vec![1]);
    assert!(null.codes.iter().all(|code| *code == 0.0));
    assert_eq!(row_loss(x.view(), &null, decoder.view(), 1), 25.0);
}

#[test]
fn scalar_budget_constructor_preserves_topk64_exactly() {
    let config = BlockSparseConfig::from_scalar_budget(114_688, 64, 4)
        .expect("DeepSeek-V3 comparison budget partitions into complete blocks");
    assert_eq!(config.n_blocks, 28_672);
    assert_eq!(config.block_topk, 16);
    assert_eq!(config.block_size, 4);
    assert_eq!(config.n_atoms(), 114_688);
    assert_eq!(config.active_atoms(), 64);
}

#[test]
fn scalar_budget_constructor_never_rounds_capacity_or_activity() {
    let capacity = BlockSparseConfig::from_scalar_budget(114_689, 64, 4)
        .expect_err("a partial final block must be rejected");
    assert!(capacity.contains("not divisible"), "{capacity}");

    let activity = BlockSparseConfig::from_scalar_budget(114_688, 63, 4)
        .expect_err("a partial active block must be rejected");
    assert!(activity.contains("not divisible"), "{activity}");

    let excessive = BlockSparseConfig::from_scalar_budget(32, 64, 4)
        .expect_err("the active budget cannot exceed scalar capacity");
    assert!(excessive.contains("active_atoms in"), "{excessive}");
}

/// Deterministic LCG in `[-1, 1)` (no RNG dependency → reproducible tests).
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / 2147483648.0) * 2.0 - 1.0
}

/// A `K×P` decoder with each block's `b` rows orthonormal (a genuine Stiefel
/// point per block), seeded pseudo-randomly.
fn make_decoder(n_blocks: usize, b: usize, p: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    let mut d = Array2::<f32>::zeros((n_blocks * b, p));
    for i in 0..n_blocks * b {
        for c in 0..p {
            d[[i, c]] = lcg(&mut s);
        }
    }
    for g in 0..n_blocks {
        let mut blk = d.slice(ndarray::s![g * b..g * b + b, ..]).to_owned();
        super::orthonormalize_block(&mut blk);
        for r in 0..b {
            for c in 0..p {
                d[[g * b + r, c]] = blk[[r, c]];
            }
        }
    }
    d
}

#[test]
fn threaded_block_router_matches_scalar_oracle_on_strided_rows_and_tile_tails() {
    let mut seed = 2826;
    let storage = Array2::from_shape_fn((38, 62), |_| lcg(&mut seed));
    let decoder_storage = Array2::from_shape_fn((34, 62), |_| lcg(&mut seed));
    let decoder = decoder_storage.slice(ndarray::s![.., ..;2]);
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("bounded routing test pool");
        pool.install(|| {
            for rows in [0, 1, 19] {
                let x = storage.slice(ndarray::s![..rows * 2;2, ..;2]);
                let expected = crate::sparse_dict::block_scoring_gpu::route_blocks_cpu(
                    x, decoder, 17, 2, 5,
                );
                for tile in [1, 5, 64] {
                    let actual = route_block_minibatch(x, decoder, 17, 2, 5, tile);
                    assert_eq!(actual.len(), expected.len());
                    for (got, want) in actual.iter().zip(&expected) {
                        assert_eq!(got.len(), want.len());
                        for (&(block, gate), &(expected_block, expected_gate)) in got.iter().zip(want) {
                            assert_eq!(block, expected_block, "threads={threads}, tile={tile}");
                            assert!(
                                (gate - expected_gate).abs() <= 32.0 * f32::EPSILON * expected_gate,
                                "gate={gate}, expected={expected_gate}, threads={threads}, tile={tile}"
                            );
                        }
                    }
                }
            }
        });
    }
}

#[test]
fn block_router_parallel_partition_preserves_wide_dictionary_routes() {
    let mut seed = 2826;
    let rows = Array2::from_shape_fn((512, 2560), |_| lcg(&mut seed));
    let decoder = Array2::from_shape_fn((2048, 2560), |_| lcg(&mut seed));
    let mut reference: Option<Vec<Vec<(u32, f32)>>> = None;
    for threads in [4, 1, 1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("bounded wide routing test pool");
        let started = std::time::Instant::now();
        let actual =
            pool.install(|| route_block_minibatch(rows.view(), decoder.view(), 1024, 2, 8, 256));
        eprintln!(
            "block router: threads={threads}, seconds={:.6}",
            started.elapsed().as_secs_f64()
        );
        assert_eq!(actual.len(), rows.nrows());
        for shortlist in &actual {
            assert_eq!(shortlist.len(), 8);
        }
        if let Some(expected) = reference.as_ref() {
            for (got, want) in actual.iter().zip(expected) {
                for (&(block, gate), &(expected_block, expected_gate)) in got.iter().zip(want) {
                    assert_eq!(block, expected_block, "threads={threads}");
                    assert!(
                        (gate - expected_gate).abs() <= 32.0 * f32::EPSILON * expected_gate,
                        "gate={gate}, expected={expected_gate}, threads={threads}"
                    );
                }
            }
        } else {
            reference = Some(actual);
        }
    }
}

/// #2634/#2825: each conditional block step sees the immediately updated
/// reconstruction and recomputes its tied codes when its frame changes.
#[test]
fn co_routed_frame_sweep_is_tied_code_descent_2634() {
    let (rows, p, b, n_blocks) = (128usize, 4usize, 2usize, 2usize);
    let mut x = Array2::<f32>::zeros((rows, p));
    for row in 0..rows {
        let theta = row as f32 * 0.19;
        let z0 = [theta.cos(), theta.sin()];
        let z1 = [(1.7 * theta).cos(), (1.7 * theta).sin()];
        x[[row, 0]] = z0[0] + 0.35 * z1[0];
        x[[row, 1]] = z0[1] + 0.35 * z1[1];
        x[[row, 2]] = 0.65 * z1[0] - 0.20 * z0[1];
        x[[row, 3]] = 0.65 * z1[1] + 0.20 * z0[0];
    }
    let mut decoder = make_decoder(n_blocks, b, p, 2634);
    let initial = route_and_code_all(
        x.view(),
        decoder.view(),
        1.0,
        n_blocks,
        b,
        n_blocks,
        rows,
        n_blocks,
    )
    .expect("initial route");
    let gamma = refresh_gamma(x.view(), &initial, decoder.view(), b);
    let codes = route_and_code_all(
        x.view(),
        decoder.view(),
        gamma,
        n_blocks,
        b,
        n_blocks,
        rows,
        n_blocks,
    )
    .expect("profiled route");
    let before = reconstruction_rss(x.view(), &codes, decoder.view(), b);
    let stationarity = refresh_frames(x.view(), &codes, &mut decoder, n_blocks, b, gamma, 0.0)
        .expect("tied frame sweep");
    let after_codes = route_and_code_all(
        x.view(),
        decoder.view(),
        gamma,
        n_blocks,
        b,
        n_blocks,
        rows,
        n_blocks,
    )
    .expect("recomputed tied codes");
    let after = reconstruction_rss(x.view(), &after_codes, decoder.view(), b);
    assert!(stationarity.is_finite() && stationarity > 1e-4);
    assert!(
        after < before,
        "one production frame sweep must decrease tied-code RSS: {before:.17e} -> {after:.17e}"
    );
}

#[test]
fn one_shot_frame_sweep_prices_both_occurrences_of_the_tied_decoder_2825() {
    let x = ndarray::array![[0.4_f32, 4.0], [0.3, -3.0], [-0.3, 3.0]];
    let mut decoder = ndarray::array![[1.0_f32, -10.0], [10.0, 3.0]];
    for mut row in decoder.outer_iter_mut() {
        let norm = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
        row.mapv_inplace(|value| (value as f64 / norm) as f32);
    }
    let initial = route_and_code_all(x.view(), decoder.view(), 1.0, 2, 1, 2, 3, 2)
        .expect("initial tied route");
    let gamma = refresh_gamma(x.view(), &initial, decoder.view(), 1);
    let codes = route_and_code_all(x.view(), decoder.view(), gamma, 2, 1, 2, 3, 2)
        .expect("profiled tied route");
    // Independent f64 projector oracle, without the stored f32 code channel.
    let x64 = x.mapv(f64::from);
    let rss = |d: &Array2<f32>| {
        let d64 = d.mapv(f64::from);
        let prediction = x64.dot(&d64.t()).dot(&d64);
        x64.iter()
            .zip(prediction.iter())
            .map(|(&value, &projected)| (value - gamma as f64 * projected).powi(2))
            .sum::<f64>()
    };
    let before = rss(&decoder);
    let stationarity = refresh_frames(x.view(), &codes, &mut decoder, 2, 1, gamma, 0.0)
        .expect("one-shot tied update");
    let after = rss(&decoder);
    assert!(
        before > 0.9 && stationarity > 1e-3,
        "the witness must be nonstationary"
    );
    assert!(
        after < before,
        "actual tied RSS increased: {before:e} -> {after:e}"
    );
    eprintln!(
        "#2825 one-shot tied RSS {before:.17e} -> {after:.17e}, stationarity={stationarity:e}"
    );
}

#[test]
fn selected_no_improvement_birth_restores_complete_one_shot_state_2023() {
    // Block 1 already reconstructs every row exactly. A duplicate frame in the
    // lower-index dead block 0 wins the deterministic TopK tie, so it is
    // SELECTED — the historical selection-only gate committed it forever even
    // though it changed no objective value. The transaction must reject on the
    // strict RSS/evidence gate and restore every live field.
    let x =
        Array2::<f32>::from_shape_fn((16, 2), |(_, column)| if column == 0 { 1.0 } else { 0.0 });
    let config = BlockSparseConfig {
        n_blocks: 2,
        block_size: 1,
        block_topk: 1,
        max_epochs: 4,
        minibatch: 16,
        block_tile: 2,
        frame_ridge: 0.0,
        aux_k: 1,
        matryoshka_prefix: false,
        tolerance: 0.0,
    };
    let mut decoder = Array2::<f32>::zeros((2, 2));
    decoder[[1, 0]] = 1.0;
    let mut gamma = 1.0_f32;
    let mut codes = route_and_code_all(x.view(), decoder.view(), gamma, 2, 1, 1, 16, 2)
        .expect("baseline route");
    let mut rss = reconstruction_rss(x.view(), &codes, decoder.view(), 1);
    let tss = centered_total_sum_squares(x.view());
    let mut criterion = explained_variance_from_rss(rss, tss);
    let proposal = BlockBirthProposal {
        block: 0,
        proposed_frame: ndarray::array![[1.0_f32, 0.0_f32]],
    };

    let mut candidate_decoder = decoder.clone();
    candidate_decoder
        .row_mut(0)
        .assign(&proposal.proposed_frame.row(0));
    let (_, candidate_codes) =
        route_and_close_gamma(x.view(), candidate_decoder.view(), gamma, &config, 1)
            .expect("candidate route");
    assert!(
        proposal_is_selected(&candidate_codes, 0, 1),
        "fixture must defeat a selection-only birth gate"
    );

    let decoder_before = decoder.clone();
    let gamma_before = gamma;
    let codes_before = codes.clone();
    let rss_before = rss;
    let criterion_before = criterion;
    let accepted = try_commit_block_birth(
        x.view(),
        &mut decoder,
        &mut gamma,
        &mut codes,
        &mut rss,
        &mut criterion,
        tss,
        &proposal,
        &config,
        1,
    )
    .expect("birth transaction");

    assert!(
        !accepted,
        "a zero-improvement selected birth must be rejected"
    );
    assert_eq!(decoder, decoder_before, "decoder frame was not restored");
    assert_eq!(gamma.to_bits(), gamma_before.to_bits());
    assert_eq!(rss.to_bits(), rss_before.to_bits());
    assert_eq!(criterion.to_bits(), criterion_before.to_bits());
    for (after, before) in codes.iter().zip(codes_before.iter()) {
        assert_eq!(after.blocks, before.blocks);
        assert_eq!(after.gates, before.gates);
        assert_eq!(after.codes, before.codes);
    }
}

#[test]
fn positive_rss_noise_birth_still_fails_rank_charge_2023() {
    let decoder = ndarray::array![[1.0_f32, 0.0_f32]];
    let gram = ndarray::array![[100.0_f64]];
    let margin =
        block_birth_evidence_margin(0, 1.0e-2, 100.0, 100, &gram, decoder.view(), 100, 2, 1)
            .expect("birth evidence calculation")
            .expect("fixture has positive realised rank");
    assert!(
        margin < 0.0,
        "a representable but sub-charge RSS gain must not birth a noise specialist: margin={margin}"
    );
}

/// `K×P` planted orthonormal atoms from a fixed symmetric matrix's eigenvectors
/// (distinct columns are orthonormal, so every block spans a distinct rank-`b`
/// subspace of `ℝ^P`).
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
    assert!(
        k <= p,
        "planted test needs K <= P for distinct orthonormal atoms"
    );
    let mut atoms = Array2::<f32>::zeros((k, p));
    for atom in 0..k {
        let col = evecs.column(atom);
        for c in 0..p {
            atoms[[atom, c]] = col[c] as f32;
        }
    }
    atoms
}

/// Data whose every row lies in exactly ONE planted block's rank-`b` subspace.
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
        for r in 0..b {
            coeffs[r] = lcg(&mut s) + 0.5; // keep away from exactly zero
        }
        for c in 0..p {
            let mut acc = 0.0f32;
            for r in 0..b {
                acc += coeffs[r] * planted[[t * b + r, c]];
            }
            x[[i, c]] = acc;
        }
    }
    x
}

#[test]
fn presence_gate_and_within_block_amplitude_are_separate() {
    let (n_blocks, b, p) = (4usize, 2usize, 6usize);
    let decoder = make_decoder(n_blocks, b, p, 77);
    let mut s = 5u64;
    let row: Array1<f32> = (0..p).map(|_| lcg(&mut s)).collect();

    let w = block_projections_row(row.view(), decoder.view(), n_blocks, b);
    let gates = block_gates(w.view());
    // Presence gate is EXACTLY the ℓ₂ norm of the within-block code (here γ = 1).
    for g in 0..n_blocks {
        let code_norm = (0..b).map(|r| w[[g, r]] * w[[g, r]]).sum::<f32>().sqrt();
        assert!(
            (gates[g] - code_norm).abs() <= 1.0e-5 * (1.0 + code_norm),
            "gate (presence) must equal the within-block code norm"
        );
    }
    // Amplitude carries DIRECTION, not just a scalar: a generic b=2 block populates
    // both internal coordinates — presence and amplitude are not the same number.
    let g = route_row_blocks(&gates, 1)[0].0 as usize;
    let nonzero = (0..b).filter(|&r| w[[g, r]].abs() > 1.0e-6).count();
    assert!(
        nonzero >= 2,
        "within-block code must retain direction, not collapse to a scalar"
    );
}

#[test]
fn routing_is_gamma_invariant() {
    // The routing gate is γ-free (block_gates reads raw projections), so scaling
    // every gate by any positive γ leaves the block selection identical.
    let gates = vec![0.1f32, 0.5, 0.3, 0.9, 0.2];
    let s1: Vec<u32> = route_row_blocks(&gates, 3).iter().map(|x| x.0).collect();
    let scaled: Vec<f32> = gates.iter().map(|g| g * 3.3).collect();
    let s2: Vec<u32> = route_row_blocks(&scaled, 3).iter().map(|x| x.0).collect();
    assert_eq!(s1, s2, "block-TopK selection must be scale (γ) invariant");
}

#[test]
fn splitting_dynamics_theorem_group_l2_kills_splitting_gradient() {
    fn group_l2_penalty(blocks: &[&[f64]], lambda: f64) -> f64 {
        lambda
            * blocks
                .iter()
                .map(|block| block.iter().map(|value| value * value).sum::<f64>().sqrt())
                .sum::<f64>()
    }

    fn l1_penalty(values: &[f64], lambda: f64) -> f64 {
        lambda * values.iter().map(|value| value.abs()).sum::<f64>()
    }

    fn log2_choose(n: usize, k: usize) -> f64 {
        assert!(k <= n, "cannot choose {k} items from {n}");
        let kk = k.min(n - k);
        (1..=kk)
            .map(|i| ((n + 1 - i) as f64 / i as f64).ln())
            .sum::<f64>()
            / std::f64::consts::LN_2
    }

    let lambda = 0.17f64;
    let block_code = [3.0f64, -4.0, 12.0];
    let rotation = [[0.6f64, -0.8, 0.0], [0.8f64, 0.6, 0.0], [0.0f64, 0.0, -1.0]];
    let rotated = [
        rotation[0][0] * block_code[0]
            + rotation[0][1] * block_code[1]
            + rotation[0][2] * block_code[2],
        rotation[1][0] * block_code[0]
            + rotation[1][1] * block_code[1]
            + rotation[1][2] * block_code[2],
        rotation[2][0] * block_code[0]
            + rotation[2][1] * block_code[1]
            + rotation[2][2] * block_code[2],
    ];

    let penalty_before = group_l2_penalty(&[&block_code], lambda);
    let penalty_after = group_l2_penalty(&[&rotated], lambda);
    assert!(
        (penalty_before - penalty_after).abs() <= 1.0e-12 * (1.0 + penalty_before.abs()),
        "group-l2 penalty must be O(b)-invariant: {penalty_before} vs {penalty_after}"
    );

    let block_reconstruction = block_code;
    let singleton_reconstruction = [block_code[0], block_code[1], block_code[2]];
    for coordinate in 0..block_reconstruction.len() {
        assert!(
            (block_reconstruction[coordinate] - singleton_reconstruction[coordinate]).abs()
                <= f64::EPSILON,
            "block and singleton decompositions must have equal reconstruction"
        );
    }

    let n_blocks = 11usize;
    let active_blocks = 1usize;
    let block_size = block_code.len();
    let selection_weight = 0.04f64;
    let block_selection_bits = log2_choose(n_blocks, active_blocks);
    let split_selection_bits = log2_choose(n_blocks * block_size, active_blocks * block_size);
    assert!(
        split_selection_bits > block_selection_bits,
        "atom-level split support catalogue must cost more bits"
    );

    let block_sparsity = group_l2_penalty(&[&block_code], lambda);
    let split_sparsity = l1_penalty(&block_code, lambda);
    assert!(
        split_sparsity > block_sparsity,
        "lasso singleton split must cost more than group-l2 for a multi-axis block"
    );

    let block_cost = block_sparsity + selection_weight * block_selection_bits;
    let split_cost = split_sparsity + selection_weight * split_selection_bits;
    assert!(
        split_cost > block_cost,
        "splitting one block into singleton atoms must strictly raise equal-fit cost: \
         block {block_cost}, split {split_cost}"
    );
}

#[test]
fn near_orthogonal_row_is_orphaned_by_gate_floor() {
    let (n_blocks, b, p, k) = (2usize, 2usize, 4usize, 2usize);
    let mut decoder = Array2::<f32>::zeros((n_blocks * b, p));
    decoder[[0, 0]] = 1.0;
    decoder[[1, 1]] = 1.0;
    decoder[[2, 0]] = std::f32::consts::FRAC_1_SQRT_2;
    decoder[[2, 1]] = std::f32::consts::FRAC_1_SQRT_2;
    decoder[[3, 0]] = -std::f32::consts::FRAC_1_SQRT_2;
    decoder[[3, 1]] = std::f32::consts::FRAC_1_SQRT_2;

    let mut x = Array2::<f32>::zeros((1, p));
    x[[0, 2]] = 1.0;

    let codes = route_and_code_all(x.view(), decoder.view(), 1.0, n_blocks, b, k, 1, 1)
        .expect("CPU block route is infallible");
    assert_eq!(codes.len(), 1);
    let code = &codes[0];
    assert!(
        code.gates.iter().all(|gate| *gate == 0.0),
        "orthogonal rows should carry only padded zero gates"
    );
    assert!(
        code.codes.iter().all(|value| *value == 0.0),
        "orthogonal rows should not populate any block code"
    );
}

#[test]
fn small_k_block_fit_runs_on_cpu_baseline_2134() {
    // #2134 wall #3: the block lane must provide a SMALL-`K` CPU baseline. The
    // device launch break-even is `n_rows·K ≥ 2^20`; a small block dictionary
    // (K = 4·2 = 8, minibatch 64 ⇒ 64·8 = 512 elems) is three orders of magnitude
    // below it, so the device could never beat the CPU here. The lane must fit it
    // on the CPU under ANY residency mode — never refuse — so the block lane can be
    // compared against the curved lane at small `K`. On this (device-absent) host
    // the CPU router runs unconditionally; the dispatch fix additionally routes a
    // below-break-even block to the exact CPU oracle on a device-PRESENT host under
    // `Required`, where it previously hard-refused. Either way, a small-`K` fit
    // must succeed and reconstruct the planted subspaces.
    let (p, b, n_blocks) = (8usize, 2usize, 4usize);
    // Break-even is n_rows·K ≥ 2^20; here minibatch·K = 64·8 = 512, far below it.
    let planted = planted_frames(p, n_blocks, b);
    let x = planted_data(&planted, n_blocks, b, p, 200);

    let config = BlockSparseConfig {
        n_blocks,
        block_size: b,
        block_topk: 1,
        max_epochs: 80,
        minibatch: 64,
        block_tile: 8,
        frame_ridge: 1.0e-9,
        aux_k: 3,
        matryoshka_prefix: false,
        tolerance: 1.0e-10,
    };
    let fit = fit_block_sparse_dictionary(x.view(), &config)
        .expect("small-K block fit must run on the CPU baseline, not refuse");
    eprintln!(
        "[#2134 small-k] EV={:.12} epochs={} convergence={:?}",
        fit.explained_variance, fit.epochs, fit.convergence
    );
    assert!(
        fit.explained_variance > 0.95,
        "small-K CPU block baseline must reconstruct the planted blocks: EV = {}",
        fit.explained_variance
    );
    // The reconstruction is the data-size N×P, computable and non-degenerate.
    let recon = fit.reconstruct();
    assert_eq!(recon.dim(), (x.nrows(), p));
    let energy: f32 = recon.iter().map(|v| v * v).sum();
    assert!(
        energy > 0.0,
        "small-K CPU baseline reconstruction must be non-trivial"
    );
}

#[test]
fn block_seed_preserves_planted_subspaces_2134() {
    // The scalar farthest-point seed used to choose G*b unrelated rows and only
    // then group adjacent pairs.  On this orthogonal four-subspace fixture that
    // produced four live mixed frames, so usage-only revival could not repair
    // the EV=0.86 local optimum.  The production block-aware seed must cover
    // every planted rank-b projector before alternating minimisation begins.
    let (p, b, n_blocks) = (8usize, 2usize, 4usize);
    let planted = planted_frames(p, n_blocks, b);
    let x = planted_data(&planted, n_blocks, b, p, 200);
    let seeded = seed_frames(x.view(), n_blocks, b);

    let projector_roundoff = (p * b * b) as f64 * f32::EPSILON as f64;
    for planted_block in 0..n_blocks {
        let mut best_overlap = f64::NEG_INFINITY;
        for seeded_block in 0..n_blocks {
            // tr(P_planted P_seeded) = ||D_planted D_seeded^T||_F^2;
            // it equals b exactly iff the two rank-b subspaces coincide.
            let mut overlap = 0.0_f64;
            for left_axis in 0..b {
                for right_axis in 0..b {
                    let mut dot = 0.0_f64;
                    for column in 0..p {
                        dot += planted[[planted_block * b + left_axis, column]] as f64
                            * seeded[[seeded_block * b + right_axis, column]] as f64;
                    }
                    overlap += dot * dot;
                }
            }
            best_overlap = best_overlap.max(overlap);
        }
        eprintln!(
            "[#2134 seed] planted_block={planted_block} best_projector_overlap={best_overlap:.12} rank={b}"
        );
        assert!(
            b as f64 - best_overlap <= projector_roundoff,
            "block-aware seed must preserve planted subspace {planted_block}: \
             best projector overlap {best_overlap} vs rank {b}"
        );
    }
}

#[test]
fn planted_block_subspaces_recovered() {
    let (p, b, n_blocks) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, n_blocks, b);
    let x = planted_data(&planted, n_blocks, b, p, 180);

    let config = BlockSparseConfig {
        n_blocks,
        block_size: b,
        block_topk: 1,
        max_epochs: 80,
        minibatch: 64,
        block_tile: 8,
        frame_ridge: 1.0e-9,
        aux_k: 3,
        matryoshka_prefix: false,
        tolerance: 1.0e-10,
    };
    let fit = fit_block_sparse_dictionary(x.view(), &config).expect("block fit");
    eprintln!(
        "[#2134 planted] EV={:.12} epochs={} convergence={:?}",
        fit.explained_variance, fit.epochs, fit.convergence
    );

    assert!(
        fit.explained_variance > 0.98,
        "planted rank-b blocks must be reconstructed: EV = {}",
        fit.explained_variance
    );

    // Every planted subspace is matched by some fitted block (small principal angle),
    // computed with the Grassmann geodesic distance from frames.rs.
    for t in 0..n_blocks {
        let mut planted_pb = Array2::<f64>::zeros((p, b));
        for r in 0..b {
            for c in 0..p {
                planted_pb[[c, r]] = planted[[t * b + r, c]] as f64;
            }
        }
        let mut best = f64::INFINITY;
        for g in 0..n_blocks {
            let mut fit_pb = Array2::<f64>::zeros((p, b));
            for r in 0..b {
                for c in 0..p {
                    fit_pb[[c, r]] = fit.decoder[[g * b + r, c]] as f64;
                }
            }
            let frame = GrassmannFrame::polar_update(fit_pb.view()).expect("fitted frame");
            let ang = frame
                .max_principal_angle(planted_pb.view())
                .expect("principal angle");
            best = best.min(ang);
        }
        eprintln!("[#2134 planted] block={t} min_principal_angle={best:.12}");
        assert!(
            best < 2.0e-2,
            "planted subspace {t} not recovered by any fitted block: min angle {best} rad"
        );
    }
}

#[test]
fn fitted_block_frames_are_orthonormal() {
    let (p, b, n_blocks) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, n_blocks, b);
    let x = planted_data(&planted, n_blocks, b, p, 120);
    let config = BlockSparseConfig {
        n_blocks,
        block_size: b,
        block_topk: 1,
        max_epochs: 40,
        minibatch: 64,
        block_tile: 8,
        frame_ridge: 1.0e-9,
        aux_k: 2,
        matryoshka_prefix: false,
        tolerance: 1.0e-9,
    };
    let fit = fit_block_sparse_dictionary(x.view(), &config).expect("block fit");
    // D_g D_gᵀ = I_b for every block.
    for g in 0..n_blocks {
        for r1 in 0..b {
            for r2 in 0..b {
                let mut dot = 0.0f32;
                for c in 0..p {
                    dot += fit.decoder[[g * b + r1, c]] * fit.decoder[[g * b + r2, c]];
                }
                let want = if r1 == r2 { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() < 1.0e-4,
                    "block {g} frame not orthonormal: <row{r1},row{r2}> = {dot}"
                );
            }
        }
    }
}

#[test]
fn utilization_and_stable_rank_reported() {
    let (p, b, n_blocks) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, n_blocks, b);
    let x = planted_data(&planted, n_blocks, b, p, 150);
    let k = 1usize;
    let config = BlockSparseConfig {
        n_blocks,
        block_size: b,
        block_topk: k,
        max_epochs: 50,
        minibatch: 64,
        block_tile: 8,
        frame_ridge: 1.0e-9,
        aux_k: 2,
        matryoshka_prefix: false,
        tolerance: 1.0e-9,
    };
    let fit = fit_block_sparse_dictionary(x.view(), &config).expect("block fit");

    assert_eq!(fit.block_utilization.len(), n_blocks);
    assert_eq!(fit.block_stable_rank.len(), n_blocks);
    // Utilisations are fractions in [0,1] summing to k (each row selects k blocks).
    let total: f32 = fit.block_utilization.iter().sum();
    assert!(
        (total - k as f32).abs() < 1.0e-3,
        "utilisation fractions must sum to block_topk={k}, got {total}"
    );
    for &u in &fit.block_utilization {
        assert!(
            (0.0..=1.0 + 1.0e-6).contains(&u),
            "utilisation out of [0,1]: {u}"
        );
    }
    // Stable rank of each used block lies in [0, b]; a block used along its full
    // 2D planted subspace has stable rank meaningfully above 1.
    for &sr in &fit.block_stable_rank {
        assert!(
            (0.0..=b as f32 + 1.0e-3).contains(&sr),
            "stable rank out of [0,b]: {sr}"
        );
    }
    let max_sr = fit.block_stable_rank.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        max_sr > 1.2,
        "a block spanning a genuine 2D subspace should report stable rank > 1.2, got {max_sr}"
    );
}

#[test]
fn block_seed_manifest_is_rust_owned_and_gauge_shaped() {
    let n = 24usize;
    let mut x = Array2::<f32>::zeros((n, 2));
    let mut blocks = ndarray::Array2::<u32>::zeros((n, 2));
    let mut codes = ndarray::Array3::<f32>::zeros((n, 2, 1));
    for i in 0..n {
        let theta = i as f32 * std::f32::consts::TAU / n as f32;
        x[[i, 0]] = theta.cos();
        x[[i, 1]] = theta.sin();
        blocks[[i, 0]] = 0;
        blocks[[i, 1]] = 1;
        codes[[i, 0, 0]] = x[[i, 0]];
        codes[[i, 1, 0]] = x[[i, 1]];
    }
    let decoder = ndarray::arr2(&[[1.0f32, 0.0], [0.0, 1.0]]);
    let counts = block_sparse_dictionary_firings(blocks.view(), 2).expect("firings");
    assert_eq!(counts, vec![n, n]);
    let config = BlockSeedManifestConfig {
        block_size: 1,
        block_topk: 2,
        gamma: 1.0,
        residual_target: false,
        n_basis_chart: 4,
        include_bases: true,
        name_prefix: "block".to_string(),
        block_tile: 2,
    };
    let manifest = block_sparse_dictionary_seed_manifest(
        x.view(),
        decoder.view(),
        blocks.view(),
        &[1.0, 1.0],
        &[1.0, 1.0],
        1.0,
        &config,
    )
    .expect("seed manifest");
    assert_eq!(manifest.n_blocks, 2);
    assert_eq!(manifest.blocks.len(), 2);
    assert_eq!(manifest.blocks[0].n_firings, n);
    assert_eq!(manifest.blocks[0].basis.as_ref().expect("basis").len(), 2);
    assert!(manifest.blocks[0].total_var > 0.0);
    assert_eq!(manifest.blocks[0].mdl_block.kind, "block");
    assert_eq!(manifest.blocks[0].mdl_chart.kind, "chart");
    // #P3 matched-DL report column: the curved chart charges its n_basis_chart
    // columns, the flat block its block_size columns, and the delta = flat − chart
    // reads the curved-vs-flat comparison in bits. Here n_basis_chart (4) > block_size
    // (1), so the chart carries the larger parameter charge and the delta is negative
    // (flat is the shorter code at these firings).
    let rec = &manifest.blocks[0];
    assert_eq!(rec.matched_dl_flat.coded_columns, config.block_size as i64);
    assert_eq!(
        rec.matched_dl_chart.coded_columns,
        config.n_basis_chart as i64
    );
    assert_eq!(rec.matched_dl_flat.n_firings, n as i64);
    assert!(rec.matched_dl_flat.total_dl_bits.is_finite());
    assert!(rec.matched_dl_chart.total_dl_bits.is_finite());
    assert!(
        (rec.matched_dl_delta_bits
            - (rec.matched_dl_flat.total_dl_bits - rec.matched_dl_chart.total_dl_bits))
            .abs()
            < 1e-9
    );
    assert!(
        rec.matched_dl_delta_bits <= 0.0,
        "flat (1 col) must be no costlier than the 4-column chart at equal firings: {}",
        rec.matched_dl_delta_bits
    );
    let recon = reconstruct_block_sparse_rows(decoder.view(), blocks.view(), codes.view(), 1)
        .expect("block reconstruct");
    for i in 0..n {
        for c in 0..2 {
            assert!((recon[[i, c]] - x[[i, c]]).abs() < 1.0e-6);
        }
    }
}

#[test]
fn block_coordinate_chart_pair_screen_accepts_split_circle() {
    let n = 96usize;
    let mut x = Array2::<f32>::zeros((n, 2));
    let mut blocks = ndarray::Array2::<u32>::zeros((n, 2));
    let mut codes = ndarray::Array3::<f32>::zeros((n, 2, 1));
    for i in 0..n {
        let theta = i as f32 * std::f32::consts::TAU / n as f32;
        x[[i, 0]] = theta.cos();
        x[[i, 1]] = theta.sin();
        blocks[[i, 0]] = 0;
        blocks[[i, 1]] = 1;
        codes[[i, 0, 0]] = x[[i, 0]];
        codes[[i, 1, 0]] = x[[i, 1]];
    }
    let decoder = ndarray::arr2(&[[1.0f32, 0.0], [0.0, 1.0]]);
    let config = BlockChartComposeConfig {
        block_size: 1,
        block_topk: 2,
        gamma: 1.0,
        residual_target: false,
        min_firings: 8,
        max_blocks: 2,
        crossfit_folds: 4,
        min_effect: 0.0,
        whitening_ridge: 1.0e-8,
        pair_screen: true,
        pair_top_blocks: 2,
        max_pairs: 1,
        pair_min_cofirings: 8,
        pair_min_score: 0.0,
        block_tile: 2,
    };
    let result = compose_block_coordinate_charts(
        x.view(),
        decoder.view(),
        blocks.view(),
        codes.view(),
        &config,
    )
    .expect("compose charts");
    assert_eq!(result.selected_chart_pairs, vec![(0, 1)]);
    assert_eq!(result.pair_records.len(), 1);
    assert!(result.pair_records[0].screen_score > 0.9);
    assert!(result.pair_records[0].evidence.deviance_gain > 0.0);
    for i in 0..n {
        let radius =
            (result.reconstructed[[i, 0]].powi(2) + result.reconstructed[[i, 1]].powi(2)).sqrt();
        assert!((radius - 1.0).abs() < 5.0e-2);
    }
}

#[test]
fn coordinate_partition_frames_are_orthonormal_and_data_independent() {
    // The cheap large-K seed is a valid St(b,P) block dictionary that never reads
    // the corpus: each block is `b` distinct signed unit coordinate axes, so the
    // within-block rows are orthonormal, and the frames depend only on (G,b,P).
    let (g, b, p) = (7usize, 3usize, 8usize);
    let frames = coordinate_partition_frames(g, b, p);
    assert_eq!(frames.dim(), (g * b, p));
    // Determinism: same shape → identical frames (fixed splitmix64 stream).
    let again = coordinate_partition_frames(g, b, p);
    assert_eq!(frames, again);
    for block in 0..g {
        for left in 0..b {
            for right in 0..b {
                let mut dot = 0.0f64;
                for c in 0..p {
                    dot += frames[[block * b + left, c]] as f64
                        * frames[[block * b + right, c]] as f64;
                }
                let want = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() < 1e-6,
                    "block {block} axes ({left},{right}) not orthonormal: {dot}"
                );
            }
        }
        // Every axis is a single signed coordinate (one non-zero entry of ±1).
        for axis in 0..b {
            let row = frames.row(block * b + axis);
            let nonzero = row.iter().filter(|&&v| v != 0.0).count();
            assert_eq!(nonzero, 1, "axis must be a single signed coordinate");
            assert!(row.iter().all(|&v| v == 0.0 || v.abs() == 1.0));
        }
    }
}

#[test]
fn data_row_seeded_entry_matches_default_byte_for_byte_2023() {
    // `fit_block_sparse_dictionary` must be exactly the DataRows case of the
    // seeded entry — same seed, same alternation, same fixed point.
    let (p, b, n_blocks) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, n_blocks, b);
    let x = planted_data(&planted, n_blocks, b, p, 180);
    let config = BlockSparseConfig {
        n_blocks,
        block_size: b,
        block_topk: 1,
        max_epochs: 40,
        minibatch: 64,
        block_tile: 8,
        frame_ridge: 1.0e-9,
        aux_k: 2,
        matryoshka_prefix: false,
        tolerance: 1.0e-10,
    };
    let default_fit = fit_block_sparse_dictionary(x.view(), &config).expect("default fit");
    let seeded_fit =
        fit_block_sparse_dictionary_with_seed(x.view(), &config, BlockSeedPolicy::DataRows)
            .expect("data-row seeded fit");
    assert_eq!(
        default_fit.decoder, seeded_fit.decoder,
        "the default entry must be byte-identical to the data-row seeded entry"
    );
    assert_eq!(
        default_fit.explained_variance,
        seeded_fit.explained_variance
    );
    assert_eq!(default_fit.epochs, seeded_fit.epochs);
}

#[test]
fn data_row_seed_places_every_overcomplete_block_in_the_observed_cloud_2023() {
    // A rotated rank-one cloud is adversarial to coordinate axes.  Even with
    // K > P, the production seed must place every scalar block on the observed
    // direction rather than create dead blocks and hope AuxK later revives them.
    let direction = [1.0_f32, 2.0, 3.0, 4.0];
    let x = Array2::from_shape_fn((32, direction.len()), |(row, column)| {
        (row as f32 - 15.5) * direction[column]
    });
    let frames = data_row_frames(x.view(), 16, 1);
    let direction_norm = direction
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    for block in 0..16 {
        let alignment = frames
            .row(block)
            .iter()
            .zip(direction.iter())
            .map(|(&left, &right)| left * right / direction_norm)
            .sum::<f32>()
            .abs();
        assert!(
            (alignment - 1.0).abs() <= 8.0 * f32::EPSILON,
            "block {block} was not seeded from the observed cloud: alignment={alignment}"
        );
    }
}

#[test]
fn coordinate_partition_seed_fits_end_to_end() {
    // The cheap large-K seed produces a valid, converged block fit on real
    // structure: it must run end to end (no seeder corpus pass) and explain a
    // non-trivial fraction of the variance. It is NOT claimed to match the
    // data-aware farthest-point seed on this adversarial orthogonal fixture — the
    // coordinate seed is the K≫intrinsic-rank front door where atoms are spurious
    // and revival, not the seed, carries recovery.
    let (p, b, n_blocks) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, n_blocks, b);
    let x = planted_data(&planted, n_blocks, b, p, 180);
    let config = BlockSparseConfig {
        n_blocks,
        block_size: b,
        block_topk: 1,
        max_epochs: 120,
        minibatch: 64,
        block_tile: 8,
        frame_ridge: 1.0e-9,
        aux_k: 3,
        matryoshka_prefix: false,
        tolerance: 1.0e-10,
    };
    let fit = fit_block_sparse_dictionary_with_seed(
        x.view(),
        &config,
        BlockSeedPolicy::CoordinatePartition,
    )
    .expect("coordinate-partition seeded fit must run end to end");
    eprintln!(
        "[#2023 coord-seed] EV={:.12} epochs={} convergence={:?}",
        fit.explained_variance, fit.epochs, fit.convergence
    );
    assert!(
        fit.explained_variance.is_finite() && fit.explained_variance > 0.5,
        "coordinate-seeded fit must explain non-trivial variance: EV = {}",
        fit.explained_variance
    );
    let recon = fit.reconstruct();
    assert_eq!(recon.dim(), (x.nrows(), p));
    assert!(recon.iter().map(|v| v * v).sum::<f32>() > 0.0);
}

#[test]
fn packed_block_gates_use_stored_codes_and_preserve_padding_2825() {
    let decoder = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
    let row = ndarray::array![3.0_f32, 4.0];
    let code = code_row(row.view(), decoder.view(), 0.7, 1, 2, &[(0, 3.0)]);
    let (blocks, gates, codes) = pack_block_codes(&[code], 2, 1);
    assert_eq!(blocks[[0, 0]], 0);
    assert_eq!(blocks[[0, 1]], 0);
    assert_eq!(gates[[0, 0]].to_bits(), codes[[0, 0, 0]].abs().to_bits());
    assert!(gates[[0, 0]] > 0.0);
    assert_eq!(gates[[0, 1]], 0.0);
    assert_eq!(codes[[0, 1, 0]], 0.0);
}

#[test]
fn fitted_block_padding_agrees_with_the_public_transform_2825() {
    let seed = coordinate_partition_frames(2, 1, 2);
    assert_eq!(seed.row(0).dot(&seed.row(1)), 0.0);
    let x = Array2::from_shape_fn((2, 2), |(row, column)| {
        (if row == 0 { 3.0 } else { -3.0 }) * seed[[0, column]]
    });
    let mut config = BlockSparseConfig::new(2, 1);
    config.block_topk = 2;
    let fit = fit_block_sparse_dictionary_with_seed(
        x.view(),
        &config,
        BlockSeedPolicy::CoordinatePartition,
    )
    .expect("the unused orthogonal block is quiescent on this rank-one corpus");
    let (blocks, gates, codes) = block_sparse_dictionary_transform(
        x.view(),
        fit.decoder.view(),
        fit.gamma,
        1,
        2,
        config.block_tile,
    )
    .unwrap();
    assert_eq!(fit.blocks, blocks);
    assert_eq!(fit.gates, gates);
    assert_eq!(fit.codes, codes);
    for row in 0..x.nrows() {
        assert_eq!(fit.blocks[[row, 0]], 0);
        assert_eq!(fit.blocks[[row, 1]], 0);
        assert_eq!(fit.gates[[row, 0]], 3.0);
        assert_eq!(fit.gates[[row, 1]], 0.0);
        assert_eq!(fit.codes[[row, 1, 0]], 0.0);
    }
}

#[test]
fn projector_distance_resolves_tiny_rotations_without_overlap_cancellation_2825() {
    let current = ndarray::array![[1.0_f32, 0.0]];
    for angle in [1.0e-5_f32, 1.0e-8, 1.0e-10] {
        let next = ndarray::array![[1.0_f32, angle]];
        let delta = angle as f64;
        // Independent two-by-two ambient projector identity. The off-diagonal
        // entries change by delta and the second diagonal by delta squared.
        let expected =
            ((2.0 * delta * delta + delta.powi(4)) / (1.0 + (1.0 + delta * delta).powi(2))).sqrt();
        let measured = frame_fixed_point_residual(current.view(), next.view(), 1, 1).unwrap();
        assert!(
            measured > 0.0,
            "a representable rotation must not collapse to zero"
        );
        assert!((measured - expected).abs() <= 32.0 * f64::EPSILON * expected);
        let negated = next.mapv(|value| -value);
        let gauge_changed =
            frame_fixed_point_residual(current.view(), negated.view(), 1, 1).unwrap();
        assert_eq!(measured, gauge_changed);
    }
    assert_eq!(
        frame_fixed_point_residual(current.view(), current.view(), 1, 1).unwrap(),
        0.0
    );
}

#[test]
fn tied_frame_stationarity_separates_normal_storage_error_from_tangent_signal_2825() {
    let current = ndarray::array![[0.6_f32, 0.8, 0.0]];
    let mut proposal = Array2::zeros((1, 3));
    let second = Array2::zeros((1, 1));
    for tangent in [0.0, 0.25] {
        let mut action = ndarray::array![[1.8_f64], [2.4], [tangent]];
        // Use the actual stored frame for an exactly normal action. Its norm
        // differs from one after f32 serialization, which must not create a
        // spurious tangent component.
        action[[0, 0]] = 3.0 * current[[0, 0]] as f64;
        action[[1, 0]] = 3.0 * current[[0, 1]] as f64;
        let scale = action.iter().map(|v| v * v).sum::<f64>().sqrt();
        let measured = super::super::block_frame::polar_tied_frame_step(
            current.view(),
            action.view_mut(),
            second.view(),
            0.0,
            1.0e30,
            proposal.view_mut(),
        )
        .unwrap();
        assert!((measured - tangent / scale).abs() <= 16.0 * f64::EPSILON);
    }
}
/// The tied loss `L(S) = ‖x − γ Σ_{g∈S} P_g x‖²` a support actually prices,
/// computed in f64 straight from the projectors — independent of anything
/// `code_row` accumulates, so it can adjudicate the selection rather than
/// restate it.
fn tied_loss_of_support(
    row: &[f64],
    decoder: ArrayView2<'_, f32>,
    blocks: &[usize],
    gamma: f64,
    b: usize,
) -> f64 {
    let p = row.len();
    let mut reconstruction = vec![0.0_f64; p];
    for &block in blocks {
        for axis in 0..b {
            let atom = decoder.row(block * b + axis);
            let mut projection = 0.0_f64;
            for (value, &direction) in row.iter().zip(atom.iter()) {
                projection += value * direction as f64;
            }
            for (out, &direction) in reconstruction.iter_mut().zip(atom.iter()) {
                *out += projection * direction as f64;
            }
        }
    }
    row.iter()
        .zip(reconstruction.iter())
        .map(|(&observed, &fitted)| {
            let residual = observed - gamma * fitted;
            residual * residual
        })
        .sum()
}

/// The blocks a row actually admitted, in slot order, skipping canonical padding.
fn admitted_blocks_of(code: &RowBlockCode) -> Vec<usize> {
    code.blocks
        .iter()
        .enumerate()
        .filter(|&(slot, _)| code.gates[slot] != 0.0)
        .map(|(_, &block)| block as usize)
        .collect()
}

/// #2825 as a PROPERTY, over an ensemble rather than a hand-built row.
/// `8aa65d500` pins the admission rule with three examples — one overlapping
/// pair refused, one orthogonal pair admitted, one non-descent scale kept
/// definable. Each is exact and each is a single row. This asserts the two
/// things that have to hold on every row of an over-complete dictionary, over a
/// deterministic ensemble of 8 blocks of `b = 2` in `P = 6` (so `K = 16 > P` and
/// the projectors overlap by construction), 64 rows at four scales:
///
/// 1. **The stopping rule is a local minimum.** At the returned support, every
///    refused candidate must price at least as much as the support without it.
///    That is a theorem about the rule, not a fact about the data, and it is
///    recomputed here from the projectors — never from anything `code_row`
///    accumulated, so the check cannot restate the thing it is checking.
/// 2. **The admitted support never prices worse than the top-`k` quota** it
///    replaces, measured on this ensemble.
///
/// Both scales below are inside `(0, 2)`, where the admission weight `2γ−γ²` is
/// positive, so the unconditional first admission that `8aa65d500` uses to keep
/// the scale definable at `γ ≥ 2` is inert here and the rule under test is the
/// conditional one throughout.
///
/// Two positive controls keep it from passing vacuously: the quota must actually
/// be reduced somewhere (or nothing is being refused), and it must be strictly
/// beaten somewhere (or the inequality is trivially satisfied by an unchanged
/// rule).
#[test]
fn greedy_admission_never_prices_worse_than_the_topk_quota_2825() {
    let p = 6usize;
    let b = 2usize;
    let n_blocks = 8usize; // K = 16 > P: over-complete, so the projectors overlap
    let k = 4usize;
    let mut decoder = Array2::<f32>::zeros((n_blocks * b, p));
    let mut state = 0x2825_u64;
    let mut next = || {
        state = splitmix64_block(state);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    for block in 0..n_blocks {
        let mut frame = Array2::<f32>::zeros((b, p));
        for axis in 0..b {
            for column in 0..p {
                frame[[axis, column]] = next() as f32;
            }
        }
        orthonormalize_block(&mut frame);
        decoder
            .slice_mut(ndarray::s![block * b..(block + 1) * b, ..])
            .assign(&frame);
    }

    let mut stuck = 0usize;
    let mut strictly_better = 0usize;
    let mut worse = 0usize;
    let mut refused_rows = 0usize;
    let rows = 64usize;
    for _ in 0..rows {
        let mut row = Array1::<f32>::zeros(p);
        for column in 0..p {
            row[column] = next() as f32;
        }
        let exact: Vec<f64> = row.iter().map(|&value| value as f64).collect();
        let projections = block_projections_row(row.view(), decoder.view(), n_blocks, b);
        let gates = block_gates(projections.view());
        let shortlist = route_row_blocks(&gates, k);
        for &gamma in &[0.4_f32, 0.8, 1.0, 1.3] {
            let code = code_row(row.view(), decoder.view(), gamma, b, k, &shortlist);
            let admitted = admitted_blocks_of(&code);
            let quota: Vec<usize> = shortlist
                .iter()
                .filter(|&&(_, gate)| gate != 0.0)
                .map(|&(block, _)| block as usize)
                .collect();
            if admitted.len() < quota.len() {
                refused_rows += 1;
            }
            // The stopping condition is a THEOREM about the returned support, not a
            // property of this ensemble: every refused candidate must price at least
            // as much as the support without it. Recomputed here from the projectors,
            // never from anything `code_row` accumulated.
            let taken = tied_loss_of_support(&exact, decoder.view(), &admitted, gamma as f64, b);
            for &block in quota.iter() {
                if admitted.contains(&block) {
                    continue;
                }
                let mut widened = admitted.clone();
                widened.push(block);
                let widened_loss =
                    tied_loss_of_support(&exact, decoder.view(), &widened, gamma as f64, b);
                let slack = 64.0 * f64::EPSILON * widened_loss.max(taken).max(1.0e-12);
                if widened_loss + slack < taken {
                    stuck += 1;
                }
            }
            let quota_loss = tied_loss_of_support(&exact, decoder.view(), &quota, gamma as f64, b);
            // The projections are f64 accumulations of f32 inputs; a support the
            // rule declined by an amount below that resolution may measure as a
            // wash either way. Anything beyond it is a real ordering.
            let resolution = 64.0 * f64::EPSILON * quota_loss.max(taken).max(1.0e-12);
            if taken > quota_loss + resolution {
                worse += 1;
            }
            if taken + resolution < quota_loss {
                strictly_better += 1;
            }
        }
    }
    assert_eq!(
        stuck, 0,
        "the returned support must be a local minimum against single additions: no \
         refused candidate may have a negative gain at it"
    );
    assert_eq!(
        worse, 0,
        "measured over this ensemble, the admitted support never priced worse than the \
         top-k quota it replaces"
    );
    // Positive control: on an over-complete dictionary the quota DOES lose, so a
    // rule that silently degenerated back to `take(k)` would fail here rather
    // than pass the inequality vacuously.
    assert!(
        refused_rows > 0,
        "the fixture must exercise refusals; the quota was never reduced"
    );
    assert!(
        strictly_better > 0,
        "the quota must be strictly beaten somewhere, or this test is vacuous"
    );
}

/// #2825: the frame convergence bar is denominated in the resolution of the
/// frames it is read from, and this pins BOTH halves of that — the half that
/// lifts and, more importantly, the half that does not.
///
/// The block dictionary is stored as `f32`, so a frame residual computed from
/// it carries `f32` round-off. Below `f32::EPSILON` such a residual is not
/// small, it is unrepresentable, and a bar placed under it asks the frames a
/// question their own storage cannot answer. Six block fits sat exactly there:
/// at `ev_residual`, `gamma_residual`, `routing_residual` and
/// `reconstruction_residual` all exactly zero, EV = 1 to fourteen figures, no
/// births and no polar failures — genuine fixed points — with frame residuals
/// of `1.7017e-8 .. 2.5664e-8` against configured tolerances of `1e-9` and
/// `1e-10`, i.e. bars asking for 119x and 1192x finer than the storage
/// resolution of `1.192093e-7`.
///
/// The danger in a floor is that it silently becomes a blanket relaxation. It
/// does not here, and that is what the second half of this test is for: a
/// tolerance ABOVE the floor must be honoured EXACTLY as configured, so the
/// floor can only ever lift a bar that was below the instrument's noise, never
/// loosen one that was above it.
#[test]
fn the_frame_bar_is_denominated_in_the_stored_frame_resolution_2825() {
    let resolution = super::super::block_frame::STORED_FRAME_RESOLUTION;
    assert_eq!(
        resolution,
        f64::from(f32::EPSILON),
        "the floor is the machine epsilon of the type the frames are stored in, \
         not a chosen number"
    );

    // The six fits' measured band, from the #2825 arm-deletion run. Every one is
    // below the storage resolution, which is why no amount of iterating cleared
    // the configured bar.
    let measured_band = [
        1.7016687085877477e-8_f64,
        2.1350316734503195e-8,
        2.3524703573643420e-8,
        2.4516826402228734e-8,
        2.5664090977161682e-8,
    ];
    for residual in measured_band {
        assert!(
            residual < resolution,
            "a residual the frames cannot resolve must sit below the floor; \
             got {residual} against {resolution}"
        );
    }

    // BELOW the floor: the bar lifts to the floor, so a fixed point the frames
    // cannot resolve past is admitted.
    for tolerance in [1.0e-10_f64, 1.0e-9] {
        let bar = tolerance.max(resolution);
        assert_eq!(bar, resolution, "a sub-resolution bar lifts to the floor");
        for residual in measured_band {
            assert!(
                residual > tolerance && residual <= bar,
                "residual {residual} is unreachable at tol {tolerance} and \
                 admitted at the floor {bar}"
            );
        }
    }

    // ABOVE the floor: unchanged, exactly. This is the half that keeps the floor
    // from being a blanket relaxation — a configured bar coarser than the
    // instrument's noise is the caller's, and the floor never touches it.
    for tolerance in [1.0e-6_f64, 1.0e-3, 1.0e-1] {
        assert_eq!(
            tolerance.max(resolution),
            tolerance,
            "a bar above the storage resolution must be honoured exactly as \
             configured, never loosened"
        );
    }
}

/// #2825 — THE FRAME RESIDUAL CANNOT RESOLVE BELOW THE STORED `f32` QUANTIZATION.
///
/// The dictionary is stored as `f32`. Two `f32` renderings of the SAME plane —
/// here the same subspace written in two different internal gauges — are not
/// the same bytes, so the projector distance between them is nonzero and of
/// order `f32::EPSILON`. That number is a property of the storage, not of the
/// fit: it is what the instrument reports when nothing moved. A bar below it
/// is therefore not a stricter test but an unreachable one, and it refuses an
/// exact fixed point on the same evidence it refuses a bad one — which is why
/// the configured `1e-9` / `1e-10` tolerances could never close.
///
/// The control is the other half: a real `1e-4` rotation OUT of the plane must
/// still sit far above the floor, so denominating the bar in the storage
/// resolution does not blind the criterion to a frame that actually moved.
#[test]
fn frame_residual_cannot_resolve_below_the_stored_f32_quantization_2825() {
    use gam_linalg::faer_ndarray::FaerEigh;
    let floor = super::super::block_frame::STORED_FRAME_RESOLUTION;
    // The two tolerances the #2825 fits were configured with. Both are BELOW
    // the storage resolution; that is the defect this test pins.
    let configured = [1.0e-9_f64, 1.0e-10];
    let mut checked = 0usize;
    for (p, b) in [(8usize, 2usize), (12, 3), (16, 4)] {
        // An orthonormal b-frame in R^p kept in f64, so the gauge rotation below
        // is exact to f64 and every difference the residual sees afterwards comes
        // from the f32 rendering alone.
        let mut a = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                a[[i, j]] = ((i * 5 + j * 3 + 1) % 13) as f64 - 6.0;
            }
        }
        let sym = &a + &a.t();
        let (_ev, evecs) = sym.eigh(faer::Side::Lower).expect("orthonormal seed");
        let mut u = Array2::<f64>::zeros((b, p));
        for r in 0..b {
            for c in 0..p {
                u[[r, c]] = evecs[[c, r]];
            }
        }
        // Same plane, different gauge: rotate the first two frame vectors into
        // each other. The SUBSPACE is unchanged, so a residual reading the
        // subspace in exact arithmetic would be identically zero.
        let theta = 0.7_f64;
        let mut rotated = u.clone();
        for c in 0..p {
            rotated[[0, c]] = theta.cos() * u[[0, c]] + theta.sin() * u[[1, c]];
            rotated[[1, c]] = -theta.sin() * u[[0, c]] + theta.cos() * u[[1, c]];
        }
        let stored = u.mapv(|value| value as f32);
        let gauged = rotated.mapv(|value| value as f32);
        let unmoved =
            frame_fixed_point_residual(stored.view(), gauged.view(), 1, b).expect("gauge residual");

        // NON-VACUITY: the two renderings must actually differ in bytes, or
        // `stored_projector_distance` short-circuits to an exact 0.0 and this
        // test would pass while measuring nothing.
        assert!(
            unmoved > 0.0,
            "p={p} b={b}: the two f32 renderings must differ, else the bitwise \
             short-circuit makes this vacuous"
        );
        assert!(
            unmoved <= floor,
            "p={p} b={b}: a frame that did not move reported {unmoved:.6e}, above \
             the storage floor {floor:.6e} the bar is denominated in"
        );
        for tolerance in configured {
            assert!(
                unmoved > tolerance,
                "p={p} b={b}: an UNMOVED frame reports {unmoved:.6e}, which must \
                 exceed the configured bar {tolerance:.6e} — that is why #2825's \
                 exactly-determined fits could never certify"
            );
        }

        // CONTROL: rotate the first frame vector 1e-4 rad out of the plane,
        // along an eigenvector the frame does not span.
        let angle = 1.0e-4_f64;
        let mut tilted = u.clone();
        for c in 0..p {
            tilted[[0, c]] = angle.cos() * u[[0, c]] + angle.sin() * evecs[[c, b]];
        }
        let moved = frame_fixed_point_residual(
            stored.view(),
            tilted.mapv(|value| value as f32).view(),
            1,
            b,
        )
        .expect("tilt residual");
        eprintln!(
            "[#2825 storage floor] p={p} b={b} floor={floor:.6e} unmoved={unmoved:.6e} \
             ({:.3}x floor) moved={moved:.6e} ({:.1}x floor)",
            unmoved / floor,
            moved / floor
        );
        assert!(
            moved > 100.0 * floor,
            "p={p} b={b}: a 1e-4 rotation reported {moved:.6e}; the floor \
             {floor:.6e} must not blind the criterion to a frame that moved"
        );
        checked += 1;
    }
    assert_eq!(checked, 3, "every shape must have been measured");
}

/// #2825 — A FIT WHOSE FRAME RESIDUAL IS STORAGE NOISE CERTIFIES.
///
/// `certified` is the field the whole convergence contract rests on and no test
/// in this crate asserted it. With the frame bar denominated in the storage
/// resolution, a planted fit closes arm (1) — and it does so with its frame
/// residual ABOVE the configured tolerance, which is the non-vacuity: it
/// certifies THROUGH the floor, not because the configured bar was met.
///
/// This test asserts only the positive, deliberately. The floor's discriminating
/// power — that it admits a frame which did not move and still refuses one that
/// did — is pinned one level down, on the residual function itself, by
/// [`frame_residual_cannot_resolve_below_the_stored_f32_quantization_2825`],
/// where a gauge-only rendering reads 0.17-0.27x the floor and a `1e-4` rotation
/// reads 419-593x it. That is where the floor is applied, so that is where the
/// separation belongs; re-deriving `tolerance.max(floor)` here would only score
/// the rule against itself.
///
/// MEASURED (this fixture, lane y5): the exactly-determined fit reports
/// `ev = 1.00000000000000`, `ev_residual = 0`, `gamma_residual = 0` and
/// `frame_residual = 2.148690e-8` — 0.180x the floor, and 215x the configured
/// `1e-10` it could never have cleared.
///
/// The over-complete arm is asserted as a SECOND POSITIVE, not as a control. It
/// was written as a control on the belief that a `K > rank` fit cannot reach a
/// frame fixed point; `8aa65d500` refuted that by making the support step a
/// descent step on the shared objective, and
/// `tiered::fit::fit_tests::tiered_certifies_at_k_gg_rank_once_the_support_step_descends_2275_2825`
/// already records it. Measured here at `frame_residual = 2.352470e-8`, which is
/// one of the six residuals the landed floor cites.
#[test]
fn exactly_determined_block_fit_certifies_at_the_storage_resolution_2825() {
    let floor = super::super::block_frame::STORED_FRAME_RESOLUTION;
    let (p, b, n_blocks) = (8usize, 2usize, 3usize);
    let planted = planted_frames(p, n_blocks, b);
    let x = planted_data(&planted, n_blocks, b, p, 180);
    let config = BlockSparseConfig {
        n_blocks,
        block_size: b,
        block_topk: 1,
        max_epochs: 80,
        minibatch: 64,
        block_tile: 8,
        frame_ridge: 1.0e-9,
        aux_k: 3,
        matryoshka_prefix: false,
        tolerance: 1.0e-10,
    };
    let fit = fit_block_sparse_dictionary(x.view(), &config).expect("exactly determined block fit");
    let exact = fit.convergence;
    eprintln!(
        "[#2825 certify] exact: ev={:.14} frame_residual={:.6e} ({:.3}x floor) \
         ev_residual={:.3e} gamma_residual={:.3e} certified={}",
        fit.explained_variance,
        exact.frame_residual,
        exact.frame_residual / floor,
        exact.ev_residual,
        exact.gamma_residual,
        exact.certified
    );
    assert!(
        exact.certified,
        "an exactly-determined planted fit must close arm (1): frame_residual \
         {:.6e} against floor {floor:.6e}",
        exact.frame_residual
    );
    // NON-VACUITY: it certified THROUGH the storage floor. If the frame residual
    // had met the configured tolerance on its own, this test would say nothing
    // about the denomination.
    assert!(
        exact.frame_residual > config.tolerance,
        "the frame residual {:.6e} met the configured bar {:.6e} unaided; this \
         fixture no longer exercises the storage floor",
        exact.frame_residual,
        config.tolerance
    );
    assert!(
        exact.frame_residual <= floor,
        "certified with frame_residual {:.6e} above the floor {floor:.6e}",
        exact.frame_residual
    );

    // The over-complete fit, as a second positive: four rank-2 blocks against
    // six planted directions. Since 8aa65d500 the support step descends the
    // shared objective, so this reaches a frame fixed point too.
    let over = BlockSparseConfig {
        n_blocks: n_blocks + 1,
        ..config
    };
    let over_fit = fit_block_sparse_dictionary(x.view(), &over).expect("over-complete block fit");
    eprintln!(
        "[#2825 certify] over-complete: frame_residual={:.6e} ({:.3}x floor) certified={}",
        over_fit.convergence.frame_residual,
        over_fit.convergence.frame_residual / floor,
        over_fit.convergence.certified
    );
    assert!(
        over_fit.convergence.certified,
        "since 8aa65d500 a K > rank fit reaches a frame fixed point too; got \
         certified=false at frame_residual {:.6e}",
        over_fit.convergence.frame_residual
    );
    assert!(
        over_fit.convergence.frame_residual <= floor,
        "the over-complete residual {:.6e} must also be storage noise, not a \
         bar the floor loosened past; floor {floor:.6e}",
        over_fit.convergence.frame_residual
    );
}
