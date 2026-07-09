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

/// Deterministic LCG in `[-1, 1)` (no RNG dependency → reproducible tests).
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / 2147483648.0) * 2.0 - 1.0
}

/// A random `b×b` orthogonal matrix (Gram–Schmidt of a fixed pseudo-random seed).
fn random_orthogonal(b: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    let mut m = Array2::<f32>::zeros((b, b));
    for i in 0..b {
        for j in 0..b {
            m[[i, j]] = lcg(&mut s);
        }
    }
    super::gram_schmidt_rows(&mut m);
    m
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
fn gauge_invariant_selection_and_loss_under_block_rotation() {
    let (n_blocks, b, p) = (5usize, 2usize, 7usize);
    let mut decoder = make_decoder(n_blocks, b, p, 12345);
    let mut s = 999u64;
    let row: Array1<f32> = (0..p).map(|_| lcg(&mut s)).collect();
    let gamma = 1.7f32;
    let k = 2usize;

    let w0 = block_projections_row(row.view(), decoder.view(), n_blocks, b);
    let gates0 = block_gates(w0.view());
    let sel0: Vec<u32> = route_row_blocks(&gates0, k)
        .iter()
        .map(|&(g, _)| g)
        .collect();
    let loss0 = row_loss(row.view(), decoder.view(), &sel0, gamma, b);

    // Rotate block g_rot's basis by a random O(b): D_g <- R D_g (rows stay orthonormal).
    let g_rot = 3usize;
    let r_mat = random_orthogonal(b, 424242);
    let old_block = decoder
        .slice(ndarray::s![g_rot * b..g_rot * b + b, ..])
        .to_owned();
    for r in 0..b {
        for c in 0..p {
            let mut acc = 0.0f32;
            for j in 0..b {
                acc += r_mat[[r, j]] * old_block[[j, c]];
            }
            decoder[[g_rot * b + r, c]] = acc;
        }
    }

    let w1 = block_projections_row(row.view(), decoder.view(), n_blocks, b);
    let gates1 = block_gates(w1.view());
    let sel1: Vec<u32> = route_row_blocks(&gates1, k)
        .iter()
        .map(|&(g, _)| g)
        .collect();
    let loss1 = row_loss(row.view(), decoder.view(), &sel1, gamma, b);

    assert_eq!(
        sel0, sel1,
        "block selection must be invariant to an O(b) gauge rotation of a block basis"
    );
    for g in 0..n_blocks {
        assert!(
            (gates0[g] - gates1[g]).abs() <= 1.0e-4 * (1.0 + gates0[g].abs()),
            "gate of block {g} changed under gauge rotation: {} vs {}",
            gates0[g],
            gates1[g]
        );
    }
    assert!(
        (loss0 - loss1).abs() <= 1.0e-4 * (1.0 + loss0.abs()),
        "loss changed under gauge rotation: {loss0} vs {loss1}"
    );

    // The tied within-block code CO-ROTATES with the basis: w'_g = w_g Rᵀ
    // (equivalently the column code z_g → R z_g), while its ℓ₂ norm — the gate —
    // is unchanged. This pins the exact gauge action, not just norm invariance.
    let w0g = w0.row(g_rot).to_owned();
    let w1g = w1.row(g_rot).to_owned();
    for i in 0..b {
        let mut expect = 0.0f32;
        for j in 0..b {
            expect += r_mat[[i, j]] * w0g[j];
        }
        assert!(
            (w1g[i] - expect).abs() <= 1.0e-4 * (1.0 + expect.abs()),
            "within-block code must co-rotate as z_g -> R z_g under a basis rotation"
        );
    }
    let n0: f32 = w0g.iter().map(|v| v * v).sum::<f32>().sqrt();
    let n1: f32 = w1g.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (n0 - n1).abs() <= 1.0e-4 * (1.0 + n0),
        "within-block code norm (the gate) must be rotation-invariant"
    );
}

#[test]
fn norm_changing_block_map_changes_selection_or_loss() {
    // NEGATIVE CONTROL: the invariance is specifically to O(b), NOT to an arbitrary
    // block map. A norm-changing (non-orthogonal) map of a SELECTED block's basis
    // must change the selection or the loss — otherwise the gauge test above would
    // pass vacuously (invariant to everything).
    let (n_blocks, b, p) = (5usize, 2usize, 7usize);
    let mut decoder = make_decoder(n_blocks, b, p, 2024);
    let mut s = 13u64;
    let row: Array1<f32> = (0..p).map(|_| lcg(&mut s)).collect();
    let gamma = 1.3f32;
    let k = 2usize;

    let gates0 = block_gates(block_projections_row(row.view(), decoder.view(), n_blocks, b).view());
    let sel0: Vec<u32> = route_row_blocks(&gates0, k).iter().map(|x| x.0).collect();
    let loss0 = row_loss(row.view(), decoder.view(), &sel0, gamma, b);

    // Scale a genuinely-selected block's basis by 2 (breaks D_g D_gᵀ = I_b).
    let g = sel0[0] as usize;
    for r in 0..b {
        for c in 0..p {
            decoder[[g * b + r, c]] *= 2.0;
        }
    }

    let gates1 = block_gates(block_projections_row(row.view(), decoder.view(), n_blocks, b).view());
    let sel1: Vec<u32> = route_row_blocks(&gates1, k).iter().map(|x| x.0).collect();
    let loss1 = row_loss(row.view(), decoder.view(), &sel1, gamma, b);

    let changed = sel0 != sel1 || (loss0 - loss1).abs() > 1.0e-3 * (1.0 + loss0.abs());
    assert!(
        changed,
        "a norm-changing block map must change selection or loss — the O(b) gauge \
         test must not be invariant to arbitrary maps (loss {loss0} vs {loss1})"
    );
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
    let rotation = [
        [0.6f64, -0.8, 0.0],
        [0.8f64, 0.6, 0.0],
        [0.0f64, 0.0, -1.0],
    ];
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
    assert!(
        fit.explained_variance > 0.95,
        "small-K CPU block baseline must reconstruct the planted blocks: EV = {}",
        fit.explained_variance
    );
    // The reconstruction is the data-size N×P, computable and non-degenerate.
    let recon = fit.reconstruct();
    assert_eq!(recon.dim(), (x.nrows(), p));
    let energy: f32 = recon.iter().map(|v| v * v).sum();
    assert!(energy > 0.0, "small-K CPU baseline reconstruction must be non-trivial");
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
fn matryoshka_prefix_losses_are_monotone_and_match_truncated_readout() {
    let (p, b, n_blocks) = (8usize, 2usize, 4usize);
    let planted = planted_frames(p, n_blocks, b);
    let x = planted_data(&planted, n_blocks, b, p, 160);
    let config = BlockSparseConfig {
        n_blocks,
        block_size: b,
        block_topk: 1,
        max_epochs: 30,
        minibatch: 64,
        block_tile: 8,
        frame_ridge: 1.0e-9,
        aux_k: 2,
        matryoshka_prefix: true,
        tolerance: 1.0e-9,
    };
    let fit = fit_block_sparse_dictionary(x.view(), &config).expect("block fit");

    assert_eq!(
        fit.matryoshka_prefix_losses
            .iter()
            .map(|&(k, _)| k)
            .collect::<Vec<_>>(),
        vec![2, 4, 8],
        "MATRYOSHKA-PREFIX ladder must be log-spaced block-aligned atom prefixes"
    );
    for pair in fit.matryoshka_prefix_losses.windows(2) {
        assert!(
            pair[1].1 <= pair[0].1 + 1.0e-8,
            "prefix losses must be monotone non-increasing: {:?}",
            fit.matryoshka_prefix_losses
        );
    }

    let mut independent_best = f64::INFINITY;
    for &(k_atoms, stored_loss) in fit.matryoshka_prefix_losses.iter() {
        let prefix_decoder = fit.decoder.slice(ndarray::s![0..k_atoms, ..]);
        let (blocks, gates, codes) = block_sparse_dictionary_transform(
            x.view(),
            prefix_decoder,
            fit.gamma,
            b,
            fit.block_topk,
            config.block_tile,
        )
        .expect("truncated prefix transform");
        assert_eq!(gates.nrows(), x.nrows(), "truncated prefix gate row count");
        let recon =
            reconstruct_block_sparse_rows(prefix_decoder, blocks.view(), codes.view(), b)
                .expect("truncated prefix reconstruction");
        let mut loss = 0.0f64;
        for i in 0..x.nrows() {
            for c in 0..x.ncols() {
                let r = x[[i, c]] as f64 - recon[[i, c]] as f64;
                loss += r * r;
            }
        }
        loss /= x.nrows() as f64;
        independent_best = independent_best.min(loss);

        let read_loss = fit
            .read_loss_at_prefix(k_atoms)
            .expect("stored prefix loss must be readable");
        assert!(
            (read_loss - stored_loss).abs() <= 1.0e-12,
            "accessor must return the stored prefix loss at K={k_atoms}"
        );
        assert!(
            (read_loss - independent_best).abs() <= 1.0e-5 * (1.0 + independent_best.abs()),
            "single nested fit loss at K={k_atoms} must match independently truncated readout: \
             stored {read_loss}, independent {independent_best}"
        );
    }
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
    assert_eq!(rec.matched_dl_chart.coded_columns, config.n_basis_chart as i64);
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
        alpha: 1.0,
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
    assert_eq!(result.accepted_pairs, vec![(0, 1)]);
    assert_eq!(result.pair_records.len(), 1);
    assert!(result.pair_records[0].screen_score > 0.9);
    assert!(result.pair_records[0].evidence.deviance_gain > 0.0);
    for i in 0..n {
        let radius =
            (result.reconstructed[[i, 0]].powi(2) + result.reconstructed[[i, 1]].powi(2)).sqrt();
        assert!((radius - 1.0).abs() < 5.0e-2);
    }
}
