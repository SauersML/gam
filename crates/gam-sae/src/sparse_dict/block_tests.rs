//! Tests for the block-sparse lane. The load-bearing one is
//! [`gauge_invariant_selection_and_loss_under_block_rotation`]: rotating a
//! block's internal basis by a random `O(b)` matrix must leave every gate,
//! the block selection, and the loss unchanged — the invariance the whole
//! design rests on.

use super::*;
use crate::frames::GrassmannFrame;
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
    assert!(k <= p, "planted test needs K <= P for distinct orthonormal atoms");
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
fn planted_data(planted: &Array2<f32>, n_blocks: usize, b: usize, p: usize, n: usize) -> Array2<f32> {
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
    let sel0: Vec<u32> = route_row_blocks(&gates0, k).iter().map(|&(g, _)| g).collect();
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
    let sel1: Vec<u32> = route_row_blocks(&gates1, k).iter().map(|&(g, _)| g).collect();
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
        assert!((0.0..=1.0 + 1.0e-6).contains(&u), "utilisation out of [0,1]: {u}");
    }
    // Stable rank of each used block lies in [0, b]; a block used along its full
    // 2D planted subspace has stable rank meaningfully above 1.
    for &sr in &fit.block_stable_rank {
        assert!(
            (0.0..=b as f32 + 1.0e-3).contains(&sr),
            "stable rank out of [0,b]: {sr}"
        );
    }
    let max_sr = fit
        .block_stable_rank
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);
    assert!(
        max_sr > 1.2,
        "a block spanning a genuine 2D subspace should report stable rank > 1.2, got {max_sr}"
    );
}
