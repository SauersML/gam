//! Unit tests for the arrow-Schur solver.

use super::*;
use approx::assert_abs_diff_eq;
use ndarray::array;

/// `SparseBlockKroneckerPenaltyOp` must reproduce the dense
/// `KroneckerPenaltyOp { factor_a: G, factor_b: I_p }` on every interface
/// (matvec, gradient, diagonal, to_dense) when the sparse block set covers
/// the same `(atom, atom')` couplings — this is the equivalence that makes
/// the sparse op a drop-in replacement for the dense data Gram.
#[test]
pub(crate) fn sparse_block_kronecker_matches_dense_kronecker() {
    // Two atoms: atom 0 has m_0 = 2 basis cols (μ offset 0), atom 1 has
    // m_1 = 3 (μ offset 2). p = 2 output channels ⇒ dim_a = 5, k = 10.
    let p = 2usize;
    let dim_a = 5usize;
    let k = dim_a * p;
    // Dense G (5×5) with non-zero (0,0), (0,1), (1,0), (1,1) atom blocks.
    let g_dense = array![
        [3.0_f64, 0.5, 0.2, -0.1, 0.0],
        [0.5, 4.0, 0.0, 0.3, 0.1],
        [0.2, 0.0, 2.0, 0.4, -0.2],
        [-0.1, 0.3, 0.4, 5.0, 0.6],
        [0.0, 0.1, -0.2, 0.6, 1.5],
    ];
    let dense = KroneckerPenaltyOp {
        factor_a: g_dense.clone(),
        factor_b: Array2::<f64>::eye(p),
        global_offset: 0,
        k,
    };
    // Sparse: atom 0 block = G[0..2, 0..2], cross blocks G[0..2,2..5] and
    // its transpose, atom 1 block = G[2..5, 2..5].
    let block_00 = g_dense.slice(ndarray::s![0..2, 0..2]).to_owned();
    let block_01 = g_dense.slice(ndarray::s![0..2, 2..5]).to_owned();
    let block_10 = g_dense.slice(ndarray::s![2..5, 0..2]).to_owned();
    let block_11 = g_dense.slice(ndarray::s![2..5, 2..5]).to_owned();
    let sparse = SparseBlockKroneckerPenaltyOp {
        p,
        dim_a,
        k,
        blocks: vec![
            SparseGBlock {
                row_off: 0,
                col_off: 0,
                data: block_00,
            },
            SparseGBlock {
                row_off: 0,
                col_off: 2,
                data: block_01,
            },
            SparseGBlock {
                row_off: 2,
                col_off: 0,
                data: block_10,
            },
            SparseGBlock {
                row_off: 2,
                col_off: 2,
                data: block_11,
            },
        ],
    };

    // to_dense parity.
    let d_dense = dense.to_dense();
    let d_sparse = sparse.to_dense();
    for i in 0..k {
        for j in 0..k {
            assert!(
                (d_dense[[i, j]] - d_sparse[[i, j]]).abs() < 1e-12,
                "to_dense mismatch at ({i},{j}): {} vs {}",
                d_dense[[i, j]],
                d_sparse[[i, j]]
            );
        }
    }

    // matvec / gradient parity on an arbitrary vector.
    let x: Vec<f64> = (0..k).map(|i| 0.1 * (i as f64) - 0.3).collect();
    let mut y_dense = vec![0.0_f64; k];
    let mut y_sparse = vec![0.0_f64; k];
    dense.matvec(&x, &mut y_dense);
    sparse.matvec(&x, &mut y_sparse);
    for i in 0..k {
        assert!(
            (y_dense[i] - y_sparse[i]).abs() < 1e-12,
            "matvec mismatch at {i}: {} vs {}",
            y_dense[i],
            y_sparse[i]
        );
    }

    // diagonal parity.
    let mut diag_dense = vec![0.0_f64; k];
    let mut diag_sparse = vec![0.0_f64; k];
    dense.diagonal(&mut diag_dense);
    sparse.diagonal(&mut diag_sparse);
    for i in 0..k {
        assert!(
            (diag_dense[i] - diag_sparse[i]).abs() < 1e-12,
            "diagonal mismatch at {i}: {} vs {}",
            diag_dense[i],
            diag_sparse[i]
        );
    }

    // block parity: probe the per-atom β block ranges.
    let offsets = [0..(2 * p), (2 * p)..k];
    for id in 0..offsets.len() {
        let b = offsets[id].end - offsets[id].start;
        let mut blk_dense = Array2::<f64>::zeros((b, b));
        let mut blk_sparse = Array2::<f64>::zeros((b, b));
        dense.block(BetaBlockId(id), &offsets, &mut blk_dense);
        sparse.block(BetaBlockId(id), &offsets, &mut blk_sparse);
        for i in 0..b {
            for j in 0..b {
                assert!(
                    (blk_dense[[i, j]] - blk_sparse[[i, j]]).abs() < 1e-12,
                    "block {id} mismatch at ({i},{j})"
                );
            }
        }
    }
}

/// Hand-built dense reference for the frame-factored Gram
/// `H[(i,li,a),(j,lj,b)] = g_ij[li,lj]·(U_iᵀU_j)[a,b]`, with the variable
/// per-atom width `r_k`.
pub(crate) fn factored_reference_dense(
    ranks: &[usize],
    basis_sizes: &[usize],
    blocks: &[FactoredFrameGBlock],
) -> Array2<f64> {
    let n_atoms = ranks.len();
    let mut offsets = vec![0usize; n_atoms + 1];
    for k in 0..n_atoms {
        offsets[k + 1] = offsets[k] + basis_sizes[k] * ranks[k];
    }
    let dim = offsets[n_atoms];
    let mut h = Array2::<f64>::zeros((dim, dim));
    for blk in blocks {
        let (r_i, r_j) = (ranks[blk.atom_i], ranks[blk.atom_j]);
        let (off_i, off_j) = (offsets[blk.atom_i], offsets[blk.atom_j]);
        let (m_i, m_j) = blk.g.dim();
        for li in 0..m_i {
            for lj in 0..m_j {
                for a in 0..r_i {
                    for b in 0..r_j {
                        h[[off_i + li * r_i + a, off_j + lj * r_j + b]] +=
                            blk.g[[li, lj]] * blk.w[[a, b]];
                    }
                }
            }
        }
    }
    h
}

/// `FactoredFrameKroneckerOp` must equal its dense `g ⊗ (UᵀU)` reference on
/// every interface, with VARIABLE per-atom rank (`r_0 = 2`, `r_1 = 3`) and a
/// genuine cross-atom output factor `U_0ᵀU_1 ≠ 0`.
#[test]
pub(crate) fn factored_frame_kronecker_matches_dense_reference() {
    // Atom 0: M_0 = 2, r_0 = 2. Atom 1: M_1 = 3, r_1 = 3. dim = 4 + 9 = 13.
    let ranks = vec![2usize, 3];
    let basis_sizes = vec![2usize, 3];
    let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
    let g11 = array![[2.0_f64, 0.4, -0.2], [0.4, 5.0, 0.6], [-0.2, 0.6, 1.5]];
    let g01 = array![[0.2_f64, -0.1, 0.0], [0.3, 0.1, -0.2]];
    let g10 = g01.t().to_owned();
    // Within-atom frame factors are identity (orthonormal U); the cross
    // factor U_0ᵀU_1 (2×3) is a generic dense principal-angle matrix.
    let w00 = Array2::<f64>::eye(2);
    let w11 = Array2::<f64>::eye(3);
    let w01 = array![[0.8_f64, 0.1, -0.05], [0.0, 0.7, 0.2]];
    let w10 = w01.t().to_owned();
    let blocks = vec![
        FactoredFrameGBlock {
            atom_i: 0,
            atom_j: 0,
            g: g00.clone(),
            w: w00.clone(),
        },
        FactoredFrameGBlock {
            atom_i: 1,
            atom_j: 1,
            g: g11.clone(),
            w: w11.clone(),
        },
        FactoredFrameGBlock {
            atom_i: 0,
            atom_j: 1,
            g: g01.clone(),
            w: w01.clone(),
        },
        FactoredFrameGBlock {
            atom_i: 1,
            atom_j: 0,
            g: g10.clone(),
            w: w10.clone(),
        },
    ];
    let op = FactoredFrameKroneckerOp::new(ranks.clone(), basis_sizes.clone(), blocks.clone())
        .expect("op");
    assert_eq!(op.dim(), 13);
    let reference = factored_reference_dense(&ranks, &basis_sizes, &blocks);

    // to_dense.
    let dense = op.to_dense();
    for i in 0..13 {
        for j in 0..13 {
            assert!(
                (dense[[i, j]] - reference[[i, j]]).abs() < 1e-12,
                "to_dense mismatch at ({i},{j}): {} vs {}",
                dense[[i, j]],
                reference[[i, j]]
            );
        }
    }
    // matvec == reference·x.
    let x: Vec<f64> = (0..13).map(|i| 0.13 * (i as f64) - 0.4).collect();
    let mut y = vec![0.0_f64; 13];
    op.matvec(&x, &mut y);
    for i in 0..13 {
        let mut expect = 0.0;
        for j in 0..13 {
            expect += reference[[i, j]] * x[j];
        }
        assert!(
            (y[i] - expect).abs() < 1e-10,
            "matvec mismatch at {i}: {} vs {expect}",
            y[i]
        );
    }
    // diagonal.
    let mut diag = vec![0.0_f64; 13];
    op.diagonal(&mut diag);
    for i in 0..13 {
        assert!(
            (diag[i] - reference[[i, i]]).abs() < 1e-12,
            "diagonal mismatch at {i}"
        );
    }
    // block over each atom's β range.
    let offsets_ranges = [0..4usize, 4..13usize];
    for id in 0..2 {
        let b = offsets_ranges[id].end - offsets_ranges[id].start;
        let mut blk = Array2::<f64>::zeros((b, b));
        op.block(BetaBlockId(id), &offsets_ranges, &mut blk);
        for bi in 0..b {
            for bj in 0..b {
                let gi = offsets_ranges[id].start + bi;
                let gj = offsets_ranges[id].start + bj;
                assert!(
                    (blk[[bi, bj]] - reference[[gi, gj]]).abs() < 1e-12,
                    "block {id} mismatch at ({bi},{bj})"
                );
            }
        }
    }
}

/// Strict-generalization pin: with every `r_k = p` and `U_k = I_p` (so all
/// frame factors are identity), `FactoredFrameKroneckerOp` reproduces
/// `SparseBlockKroneckerPenaltyOp` (the `G ⊗ I_p` data Gram) bit-for-bit on
/// matvec — i.e. the full-`B` border is the `r = p` special case of the
/// factored op, not a separate path.
#[test]
pub(crate) fn factored_frame_kronecker_reduces_to_sparse_block_at_full_rank() {
    let p = 2usize;
    let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
    let g11 = array![[2.0_f64, 0.4], [0.4, 5.0]];
    let g01 = array![[0.2_f64, -0.1], [0.3, 0.1]];
    let g10 = g01.t().to_owned();
    // Factored op with r_k = p, U = I_p (w = I_p everywhere).
    let ident = Array2::<f64>::eye(p);
    let factored = FactoredFrameKroneckerOp::new(
        vec![p, p],
        vec![2, 2],
        vec![
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 0,
                g: g00.clone(),
                w: ident.clone(),
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 1,
                g: g11.clone(),
                w: ident.clone(),
            },
            FactoredFrameGBlock {
                atom_i: 0,
                atom_j: 1,
                g: g01.clone(),
                w: ident.clone(),
            },
            FactoredFrameGBlock {
                atom_i: 1,
                atom_j: 0,
                g: g10.clone(),
                w: ident.clone(),
            },
        ],
    )
    .expect("factored op");
    // Equivalent SparseBlockKroneckerPenaltyOp (μ-major / oc-minor, p=2).
    let sparse = SparseBlockKroneckerPenaltyOp {
        p,
        dim_a: 4,
        k: 8,
        blocks: vec![
            SparseGBlock {
                row_off: 0,
                col_off: 0,
                data: g00,
            },
            SparseGBlock {
                row_off: 2,
                col_off: 2,
                data: g11,
            },
            SparseGBlock {
                row_off: 0,
                col_off: 2,
                data: g01,
            },
            SparseGBlock {
                row_off: 2,
                col_off: 0,
                data: g10,
            },
        ],
    };
    assert_eq!(factored.dim(), sparse.dim());
    let x: Vec<f64> = (0..8).map(|i| 0.2 * (i as f64) - 0.5).collect();
    let mut yf = vec![0.0_f64; 8];
    let mut ys = vec![0.0_f64; 8];
    factored.matvec(&x, &mut yf);
    sparse.matvec(&x, &mut ys);
    for i in 0..8 {
        assert!(
            (yf[i] - ys[i]).abs() < 1e-12,
            "full-rank factored op must equal SparseBlockKronecker at {i}: {} vs {}",
            yf[i],
            ys[i]
        );
    }
}

/// Modified Gram–Schmidt orthonormalization of the columns of a `p × r`
/// matrix (`r ≤ p`), used by the frame-constructor tests to build genuine
/// `St(p, r)` representatives. Returns the orthonormal `Q` (`p × r`).
pub(crate) fn mgs_orthonormalize(a: &Array2<f64>) -> Array2<f64> {
    let (p, r) = a.dim();
    let mut q = a.clone();
    for j in 0..r {
        // Subtract projections onto the already-orthonormalized columns.
        for i in 0..j {
            let mut dot = 0.0;
            for c in 0..p {
                dot += q[[c, i]] * q[[c, j]];
            }
            for c in 0..p {
                q[[c, j]] -= dot * q[[c, i]];
            }
        }
        let mut nrm = 0.0;
        for c in 0..p {
            nrm += q[[c, j]] * q[[c, j]];
        }
        let nrm = nrm.sqrt();
        assert!(nrm > 1e-9, "mgs column {j} degenerate");
        for c in 0..p {
            q[[c, j]] /= nrm;
        }
    }
    q
}

/// `frame_output_gram` of an orthonormal frame with itself is the identity.
#[test]
pub(crate) fn frame_output_gram_orthonormal_is_identity() {
    let p = 5usize;
    let r = 3usize;
    // A deterministic-but-generic p×r seed, then orthonormalize.
    let mut seed = Array2::<f64>::zeros((p, r));
    for c in 0..p {
        for a in 0..r {
            seed[[c, a]] = ((c as f64) * 0.37 + (a as f64) * 1.31).sin() + 0.1 * (a as f64);
        }
    }
    let u = mgs_orthonormalize(&seed);
    let g = frame_output_gram(u.view(), u.view());
    assert_eq!(g.dim(), (r, r));
    for a in 0..r {
        for b in 0..r {
            let expect = if a == b { 1.0 } else { 0.0 };
            assert!(
                (g[[a, b]] - expect).abs() < 1e-12,
                "UᵀU not identity at ({a},{b}): {}",
                g[[a, b]]
            );
        }
    }
}

/// `from_frames_and_blocks` with two genuinely orthonormal frames must
/// reproduce the hand-built dense `g ⊗ (UᵀU)` reference on every interface,
/// computing the `W_ij` factors itself from the supplied frames.
#[test]
pub(crate) fn from_frames_and_blocks_matches_dense_reference() {
    let p = 4usize;
    // Atom 0: M_0 = 2, r_0 = 2. Atom 1: M_1 = 3, r_1 = 3.
    let basis_sizes = vec![2usize, 3];
    // Build two generic seeds and orthonormalize into St(p, r) frames.
    let mut seed0 = Array2::<f64>::zeros((p, 2));
    let mut seed1 = Array2::<f64>::zeros((p, 3));
    for c in 0..p {
        for a in 0..2 {
            seed0[[c, a]] = ((c as f64) * 0.91 - (a as f64) * 0.5).cos() + 0.2 * (c as f64);
        }
        for a in 0..3 {
            seed1[[c, a]] = ((c as f64) * 0.23 + (a as f64) * 1.7).sin() - 0.3 * (a as f64);
        }
    }
    let u0 = mgs_orthonormalize(&seed0);
    let u1 = mgs_orthonormalize(&seed1);

    let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
    let g11 = array![[2.0_f64, 0.4, -0.2], [0.4, 5.0, 0.6], [-0.2, 0.6, 1.5]];
    let g01 = array![[0.2_f64, -0.1, 0.0], [0.3, 0.1, -0.2]];
    let g10 = g01.t().to_owned();

    let mut g_blocks: std::collections::BTreeMap<(usize, usize), Array2<f64>> =
        std::collections::BTreeMap::new();
    g_blocks.insert((0, 0), g00.clone());
    g_blocks.insert((1, 1), g11.clone());
    g_blocks.insert((0, 1), g01.clone());
    g_blocks.insert((1, 0), g10.clone());

    let frames = vec![Some(u0.clone()), Some(u1.clone())];
    let op = FactoredFrameKroneckerOp::from_frames_and_blocks(&frames, &basis_sizes, p, &g_blocks)
        .expect("from_frames_and_blocks");
    // dim = M_0·r_0 + M_1·r_1 = 2·2 + 3·3 = 13.
    assert_eq!(op.dim(), 13);

    // Hand-built dense reference: W_ij = U_iᵀ U_j computed independently.
    let ranks = vec![2usize, 3];
    let w00 = frame_output_gram(u0.view(), u0.view());
    let w11 = frame_output_gram(u1.view(), u1.view());
    let w01 = frame_output_gram(u0.view(), u1.view());
    let w10 = frame_output_gram(u1.view(), u0.view());
    let ref_blocks = vec![
        FactoredFrameGBlock {
            atom_i: 0,
            atom_j: 0,
            g: g00,
            w: w00,
        },
        FactoredFrameGBlock {
            atom_i: 1,
            atom_j: 1,
            g: g11,
            w: w11,
        },
        FactoredFrameGBlock {
            atom_i: 0,
            atom_j: 1,
            g: g01,
            w: w01,
        },
        FactoredFrameGBlock {
            atom_i: 1,
            atom_j: 0,
            g: g10,
            w: w10,
        },
    ];
    let reference = factored_reference_dense(&ranks, &basis_sizes, &ref_blocks);

    let dense = op.to_dense();
    for i in 0..13 {
        for j in 0..13 {
            assert!(
                (dense[[i, j]] - reference[[i, j]]).abs() < 1e-12,
                "to_dense mismatch at ({i},{j}): {} vs {}",
                dense[[i, j]],
                reference[[i, j]]
            );
        }
    }
    // matvec == reference·x.
    let x: Vec<f64> = (0..13).map(|i| 0.17 * (i as f64) - 0.6).collect();
    let mut y = vec![0.0_f64; 13];
    op.matvec(&x, &mut y);
    for i in 0..13 {
        let mut expect = 0.0;
        for j in 0..13 {
            expect += reference[[i, j]] * x[j];
        }
        assert!(
            (y[i] - expect).abs() < 1e-10,
            "matvec mismatch at {i}: {} vs {expect}",
            y[i]
        );
    }
}

/// Mixed framed/unframed case: atom 0 framed (`r_0 = 2 < p = 4`), atom 1
/// unframed (`None → r_1 = p = 4`). The constructor must stand `I_p` in for
/// the missing frame, so the within-atom-1 block is exactly `g_11 ⊗ I_4`.
#[test]
pub(crate) fn from_frames_and_blocks_mixed_framed_unframed() {
    let p = 4usize;
    let basis_sizes = vec![2usize, 2]; // M_0 = 2, M_1 = 2.
    // Atom 0 gets a genuine orthonormal 4×2 frame; atom 1 stays full-B.
    let mut seed0 = Array2::<f64>::zeros((p, 2));
    for c in 0..p {
        for a in 0..2 {
            seed0[[c, a]] = ((c as f64) * 0.61 + (a as f64) * 0.9).cos() - 0.15 * (c as f64);
        }
    }
    let u0 = mgs_orthonormalize(&seed0);

    let g00 = array![[3.0_f64, 0.5], [0.5, 4.0]];
    let g11 = array![[2.0_f64, 0.4], [0.4, 5.0]];
    let g01 = array![[0.2_f64, -0.1], [0.3, 0.1]];
    let g10 = g01.t().to_owned();

    let mut g_blocks: std::collections::BTreeMap<(usize, usize), Array2<f64>> =
        std::collections::BTreeMap::new();
    g_blocks.insert((0, 0), g00.clone());
    g_blocks.insert((1, 1), g11.clone());
    g_blocks.insert((0, 1), g01.clone());
    g_blocks.insert((1, 0), g10.clone());

    let frames = vec![Some(u0.clone()), None];
    let op = FactoredFrameKroneckerOp::from_frames_and_blocks(&frames, &basis_sizes, p, &g_blocks)
        .expect("from_frames_and_blocks mixed");

    // dim = M_0·r_0 + M_1·r_1 = 2·2 + 2·4 = 12.
    assert_eq!(op.ranks, vec![2usize, 4]);
    assert_eq!(op.dim(), 12);

    // The within-unframed-atom block (atom 1) must be exactly g_11 ⊗ I_4.
    // Atom 1's β range starts at offset M_0·r_0 = 4 and spans M_1·r_1 = 8.
    let dense = op.to_dense();
    let off1 = 4usize;
    for li in 0..2 {
        for lj in 0..2 {
            for a in 0..4 {
                for b in 0..4 {
                    let gi = off1 + li * 4 + a;
                    let gj = off1 + lj * 4 + b;
                    let expect = if a == b { g11[[li, lj]] } else { 0.0 };
                    assert!(
                        (dense[[gi, gj]] - expect).abs() < 1e-12,
                        "g_11 ⊗ I_4 mismatch at ({gi},{gj}): {} vs {expect}",
                        dense[[gi, gj]]
                    );
                }
            }
        }
    }

    // Full dense reference: W computed with U_1 = I_p for the unframed atom.
    let ranks = vec![2usize, 4];
    let ident_p = Array2::<f64>::eye(p);
    let w00 = frame_output_gram(u0.view(), u0.view());
    let w11 = frame_output_gram(ident_p.view(), ident_p.view());
    let w01 = frame_output_gram(u0.view(), ident_p.view());
    let w10 = frame_output_gram(ident_p.view(), u0.view());
    let ref_blocks = vec![
        FactoredFrameGBlock {
            atom_i: 0,
            atom_j: 0,
            g: g00,
            w: w00,
        },
        FactoredFrameGBlock {
            atom_i: 1,
            atom_j: 1,
            g: g11.clone(),
            w: w11,
        },
        FactoredFrameGBlock {
            atom_i: 0,
            atom_j: 1,
            g: g01,
            w: w01,
        },
        FactoredFrameGBlock {
            atom_i: 1,
            atom_j: 0,
            g: g10,
            w: w10,
        },
    ];
    let reference = factored_reference_dense(&ranks, &basis_sizes, &ref_blocks);

    // matvec == reference·x.
    let x: Vec<f64> = (0..12).map(|i| 0.11 * (i as f64) - 0.4).collect();
    let mut y = vec![0.0_f64; 12];
    op.matvec(&x, &mut y);
    for i in 0..12 {
        let mut expect = 0.0;
        for j in 0..12 {
            expect += reference[[i, j]] * x[j];
        }
        assert!(
            (y[i] - expect).abs() < 1e-10,
            "mixed matvec mismatch at {i}: {} vs {expect}",
            y[i]
        );
    }
}

/// Verify the arrow-Schur solve against a small dense reference.
/// Build the joint bordered system as a single dense (K + N·d)² matrix,
/// solve it with the local cholesky_lower path, and compare to the
/// arrow-Schur output.
#[test]
pub(crate) fn arrow_schur_matches_dense_reference_2x2() {
    // N = 2 rows, d = 2 latent, K = 3 β.
    let n = 2;
    let d = 2;
    let k = 3;
    let mut sys = ArrowSchurSystem::new(n, d, k);

    // Row 0: H_tt = [[2, 0.1],[0.1, 3]], H_tβ = [[1, 0, 0.5],[0.2, 1, 0]],
    //         g_t = [0.3, -0.2].
    sys.rows[0].htt = array![[2.0_f64, 0.1], [0.1, 3.0]];
    sys.rows[0].htbeta = array![[1.0_f64, 0.0, 0.5], [0.2, 1.0, 0.0]];
    sys.rows[0].gt = array![0.3_f64, -0.2];

    // Row 1.
    sys.rows[1].htt = array![[1.5_f64, -0.1], [-0.1, 2.0]];
    sys.rows[1].htbeta = array![[0.1_f64, 0.5, 0.0], [0.0, 0.3, 1.0]];
    sys.rows[1].gt = array![-0.1_f64, 0.4];

    // β-block.
    sys.hbb = array![[4.0_f64, 0.2, 0.0], [0.2, 5.0, 0.1], [0.0, 0.1, 6.0],];
    sys.gb = array![0.5_f64, -0.3, 0.2];

    let (delta_t, delta_beta, _diag) = sys.solve(0.0, 0.0).expect("arrow-schur solve");
    let streaming_options = ArrowSolveOptions::direct().with_streaming_chunk_size(Some(1));
    let (delta_t_stream, delta_beta_stream, _diag_stream) = sys
        .solve_with_options(0.0, 0.0, &streaming_options)
        .expect("streaming arrow-schur solve");
    assert_eq!(delta_beta, delta_beta_stream);
    assert_eq!(delta_t, delta_t_stream);

    // Build dense reference: order is [β; t_0; t_1] = K + N·d entries.
    let total = k + n * d;
    let mut hjoint = Array2::<f64>::zeros((total, total));
    let mut gjoint = Array1::<f64>::zeros(total);
    // β-β block.
    for a in 0..k {
        for b in 0..k {
            hjoint[[a, b]] = sys.hbb[[a, b]];
        }
        gjoint[a] = sys.gb[a];
    }
    // t-blocks and cross-blocks.
    for i in 0..n {
        let toff = k + i * d;
        for a in 0..d {
            for b in 0..d {
                hjoint[[toff + a, toff + b]] = sys.rows[i].htt[[a, b]];
            }
            gjoint[toff + a] = sys.rows[i].gt[a];
            for a2 in 0..k {
                hjoint[[toff + a, a2]] = sys.rows[i].htbeta[[a, a2]];
                hjoint[[a2, toff + a]] = sys.rows[i].htbeta[[a, a2]];
            }
        }
    }
    // Solve hjoint · x = -gjoint via cholesky.
    let lj = cholesky_lower(&hjoint).expect("dense ref PD");
    let neg_g = gjoint.mapv(|v| -v);
    let xref = cholesky_solve_vector(&lj, &neg_g);
    // Compare β.
    for a in 0..k {
        assert!(
            (xref[a] - delta_beta[a]).abs() < 1e-10,
            "β[{a}] mismatch: dense {} vs arrow {}",
            xref[a],
            delta_beta[a]
        );
    }
    // Compare t.
    for i in 0..n {
        for a in 0..d {
            let dense = xref[k + i * d + a];
            let arrow = delta_t[i * d + a];
            assert!(
                (dense - arrow).abs() < 1e-10,
                "t[{i},{a}] mismatch: dense {dense} vs arrow {arrow}"
            );
        }
    }
}

pub(crate) fn diagonal_arrow_fixture(row_min: f64, schur_min: f64) -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(2, 2, 2);
    sys.rows[0].htt = array![[row_min, 0.0], [0.0, row_min + 1.0]];
    sys.rows[1].htt = array![[row_min + 2.0, 0.0], [0.0, row_min + 3.0]];
    for row in sys.rows.iter_mut() {
        row.htbeta.fill(0.0);
        row.gt.fill(0.0);
    }
    sys.hbb = array![[schur_min, 0.0], [0.0, schur_min + 1.0]];
    sys.gb.fill(0.0);
    sys
}

pub(crate) fn diagonal_fixture_dense_lambda_min(sys: &ArrowSchurSystem) -> f64 {
    let mut out = f64::INFINITY;
    for row in &sys.rows {
        for axis in 0..row.htt.nrows() {
            out = out.min(row.htt[[axis, axis]]);
        }
    }
    for axis in 0..sys.hbb.nrows() {
        out = out.min(sys.hbb[[axis, axis]]);
    }
    out
}

#[test]
pub(crate) fn arrow_factor_min_pivot_matches_dense_lambda_min_ordering() {
    let weak = diagonal_arrow_fixture(0.2, 0.8);
    let strong = diagonal_arrow_fixture(0.7, 1.2);
    let options = ArrowSolveOptions::direct();
    let (_dt_w, _db_w, weak_cache) =
        solve_arrow_newton_step_with_options(&weak, 0.0, 0.0, &options)
            .expect("weak diagonal fixture should factor");
    let (_dt_s, _db_s, strong_cache) =
        solve_arrow_newton_step_with_options(&strong, 0.0, 0.0, &options)
            .expect("strong diagonal fixture should factor");

    let weak_lambda = diagonal_fixture_dense_lambda_min(&weak);
    let strong_lambda = diagonal_fixture_dense_lambda_min(&strong);
    assert!(weak_lambda < strong_lambda);

    let weak_pivot = arrow_factor_min_pivot(&weak_cache)
        .min_pivot
        .expect("weak pivot");
    let strong_pivot = arrow_factor_min_pivot(&strong_cache)
        .min_pivot
        .expect("strong pivot");
    assert_abs_diff_eq!(weak_pivot, weak_lambda, epsilon = 1.0e-14);
    assert_abs_diff_eq!(strong_pivot, strong_lambda, epsilon = 1.0e-14);
    assert!(weak_pivot < strong_pivot);
}

pub(crate) fn quartic_counterexample_value(t: f64) -> f64 {
    0.25 * t.powi(4) - t * t + 2.0 * t
}

pub(crate) fn quartic_counterexample_system(t: f64) -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(1, 1, 0);
    sys.rows[0].gt = array![t.powi(3) - 2.0 * t + 2.0];
    sys.rows[0].htt = array![[3.0 * t * t - 2.0]];
    sys
}

#[test]
pub(crate) fn proximal_correction_breaks_scalar_newton_cycle() {
    let options = ArrowSolveOptions::direct();
    let correction = ArrowProximalCorrectionOptions {
        initial_ridge: 1e-8,
        ridge_growth: 10.0,
        max_attempts: 16,
        armijo_c1: 1e-4,
        gradient_tolerance: 1e-12,
        convergence_objective_rel_tol: DEFAULT_PROXIMAL_CONVERGENCE_REL_TOL,
    };
    let mut t = 0.0_f64;
    let mut previous_value = quartic_counterexample_value(t);

    for _ in 0..32 {
        let sys = quartic_counterexample_system(t);
        let accepted = solve_arrow_newton_step_with_proximal_correction(
            &sys,
            0.0,
            0.0,
            previous_value,
            &options,
            &correction,
            |delta_t, _delta_beta| quartic_counterexample_value(t + delta_t[0]),
        )
        .expect("proximal correction should accept a descent step");
        assert!(
            accepted.trial_objective_value <= previous_value,
            "accepted step must not increase the objective"
        );
        t += accepted.delta_t[0];
        previous_value = accepted.trial_objective_value;
    }

    let final_grad = t.powi(3) - 2.0 * t + 2.0;
    assert!(
        final_grad.abs() < 1e-7,
        "corrected iteration should reach the scalar critical point; t={t}, g={final_grad}"
    );
}

/// Issue #195 / gam#578: a per-row block that is barely-PD (smallest
/// pivot on the order of ε·trace — a rank-deficient / over-parameterized
/// decoder atom) factors successfully but is unsafe to use raw in the
/// Schur reduction. The κ proxy is folded INTO the per-row ridge
/// escalation loop: rather than reject such a block outright (which made
/// the advertised Arrow-Schur ridge never actually run and aborted the
/// whole SAE fit, gam#578), `factor_one_row` lifts this row's ridge until
/// the block is BOTH positive-definite and well-conditioned, then returns
/// a genuinely conditioned factor safe to plug into
/// `S = H_ββ + ridge_β·I − Σ_i H_tβ^(i)ᵀ (H_tt^(i))⁻¹ H_tβ^(i)`.
/// Only a block that cannot be conditioned even at `ridge_cap` errors.
#[test]
pub(crate) fn factor_one_row_conditions_barely_pd_block_via_ridge() {
    let d = 2;
    let k = 2;
    let mut row = ArrowRowBlock::new(d, k);
    // Matrix from the issue body: PD by an exact ε along the second
    // direction. Cholesky succeeds at ridge 0, but κ ≈ 1e14 — far past
    // the safe inversion regime. This is exactly the rank-deficient
    // decoder-atom block gam#578 advertised the ridge would stabilize.
    row.htt = array![[1.0_f64, 1.0], [1.0, 1.0 + 1e-14]];
    row.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
    row.gt = array![0.0_f64, 0.0];

    // The fix: instead of rejecting, the escalation loop lifts this
    // row's ridge until the factor is well-conditioned. The returned
    // factor must satisfy the κ ceiling that a raw barely-PD block fails.
    let factor = factor_one_row(&row, 0.0, d, 0, false).expect(
        "barely-PD H_tt must be CONDITIONED by per-row ridge escalation, not rejected (gam#578)",
    );
    let kappa = cholesky_factor_kappa_estimate(&factor);
    assert!(
        kappa.is_finite() && kappa <= safe_spd_kappa_max(d),
        "conditioned factor must be within the safe-inversion κ ceiling; got κ={kappa:e}"
    );
    // The factor is a genuine Cholesky of the ridge-lifted block
    // H_tt + ridge_eff·I (ridge_eff ≥ 0), so reconstructing L Lᵀ must
    // match H_tt up to a nonnegative diagonal shift (never below).
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0_f64;
            for kk in 0..d {
                acc += factor[[i, kk]] * factor[[j, kk]];
            }
            if i == j {
                assert!(
                    acc >= row.htt[[i, j]] - 1e-12,
                    "diagonal of L Lᵀ must be H_tt + (nonneg ridge) at ({i},{j}): \
                         {acc} vs {}",
                    row.htt[[i, j]]
                );
            } else {
                assert!(
                    (acc - row.htt[[i, j]]).abs() < 1e-9,
                    "off-diagonal of L Lᵀ must equal H_tt at ({i},{j}): {acc} vs {}",
                    row.htt[[i, j]]
                );
            }
        }
    }

    // Evidence/log-det mode (`tolerate_ill_conditioning = true`) must
    // accept the same barely-PD block and return its genuine Cholesky
    // factor — the diagonal gives an exact log-determinant.
    let factor = factor_one_row(&row, 0.0, d, 0, true)
        .expect("tolerate_ill_conditioning must accept a barely-PD-but-PD block");
    // L Lᵀ must reproduce the original block (the factor is real, not a
    // damped surrogate).
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0_f64;
            for kk in 0..d {
                acc += factor[[i, kk]] * factor[[j, kk]];
            }
            assert!(
                (acc - row.htt[[i, j]]).abs() < 1e-12,
                "tolerated factor must satisfy L Lᵀ = H_tt at ({i},{j})"
            );
        }
    }

    // A genuinely non-PD block must STILL error even under tolerance —
    // the flag lifts only the κ rejection, not the PD requirement.
    let mut row_npd = ArrowRowBlock::new(d, k);
    row_npd.htt = array![[1.0_f64, 2.0], [2.0, 1.0]]; // indefinite (eigvals 3, -1)
    row_npd.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
    row_npd.gt = array![0.0_f64, 0.0];
    let npd = factor_one_row(&row_npd, 0.0, d, 0, true);
    assert!(
        matches!(npd, Err(ArrowSchurError::PerRowFactorFailed { .. })),
        "non-PD block must error even with tolerate_ill_conditioning; got {npd:?}"
    );

    // Sanity: a well-conditioned block at the same dimension still
    // factors successfully.
    let mut row_ok = ArrowRowBlock::new(d, k);
    row_ok.htt = array![[2.0_f64, 0.1], [0.1, 3.0]];
    row_ok.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
    row_ok.gt = array![0.0_f64, 0.0];
    factor_one_row(&row_ok, 0.0, d, 0, false)
        .expect("well-conditioned block must still factor at ridge_t=0");

    // A block that cannot be conditioned at all — a non-finite entry —
    // is genuinely broken: no finite ridge shift repairs it, so the
    // escalation loop must still surface a typed `PerRowFactorFailed`
    // for the outer loop rather than loop forever or return garbage.
    let mut row_nan = ArrowRowBlock::new(d, k);
    row_nan.htt = array![[f64::NAN, 0.0], [0.0, 1.0]];
    row_nan.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
    row_nan.gt = array![0.0_f64, 0.0];
    let nan = factor_one_row(&row_nan, 1.0e-6, d, 0, false);
    assert!(
        matches!(nan, Err(ArrowSchurError::PerRowFactorFailed { .. })),
        "non-finite block must surface PerRowFactorFailed, not loop or condition; got {nan:?}"
    );
}

#[test]
pub(crate) fn factor_one_row_conditions_scalar_tiny_pivot_via_ridge() {
    let d = 1;
    let k = 1;
    let mut row = ArrowRowBlock::new(d, k);
    row.htt = array![[1.0e-20_f64]];
    row.htbeta = array![[1.0_f64]];
    row.gt = array![0.0_f64];

    let factor = factor_one_row(&row, 0.0, d, 0, false)
        .expect("tiny positive scalar pivot must be ridge-conditioned");
    let pivot = factor[[0, 0]] * factor[[0, 0]];
    assert!(
        pivot >= safe_spd_pivot_min(1.0),
        "scalar pivot must be lifted above the absolute safe floor; got {pivot:e}"
    );
    assert!(
        pivot > row.htt[[0, 0]],
        "scalar block must not be accepted at the raw tiny pivot"
    );

    let tolerated = factor_one_row(&row, 0.0, d, 0, true)
        .expect("tolerated log-det path must accept a positive scalar block");
    let raw_pivot = tolerated[[0, 0]] * tolerated[[0, 0]];
    assert!(
        (raw_pivot - row.htt[[0, 0]]).abs() < 1.0e-30,
        "tolerated factor must remain the raw scalar Cholesky"
    );
}

/// #1117/#1118: a per-row `H_tt` that is gauge-flat AND genuinely indefinite
/// off the gauge orbit (the K>1 IBP/softmax row-sharing state) must be
/// conditioned by the undamped evidence factor through **unit-stiffness
/// spectral deflation** — `factor_spectral_deflated_evidence_row` discovers
/// the negative/flat eigen-direction the closed-form gauge deflation cannot
/// rescue and stiffens it to eigenvalue `+1` (a ρ-independent `log 1 = 0`
/// evidence contribution), NOT a ρ-dependent `+ridge·I` bias. And the
/// STATIONARY version of the same block (the indefinite direction now
/// positive, i.e. genuinely PD) must factor through the undamped evidence
/// path to the EXACT Cholesky `L Lᵀ = H_tt` with NO bias. This pins the
/// contract the `converge_inner_for_undamped_logdet` path relies on:
/// finite-and-bias-free pre-stationarity (so the outer REML value and its
/// analytic ρ-gradient agree), exact-and-unbiased at the optimum.
#[test]
pub(crate) fn evidence_row_spectral_deflates_indefinite_non_gauge_block_at_unit_stiffness() {
    let d = 3usize;
    let k = 2usize;

    // Pre-stationarity block: e_1 is a near-null GAUGE direction (curvature
    // 1e-10, far below GAUGE_RAYLEIGH_EPS·max_diag = 1e-8·4 = 4e-8, so it
    // qualifies for Faddeev-Popov deflation), e_2 is GENUINELY indefinite
    // (eigenvalue −1.0 — real negative curvature, NOT a gauge orbit). The
    // gauge deflation lifts only e_1 (→ +1), leaving the −1.0 along e_2, so
    // the closed-form gauge deflation alone cannot make the block PD.
    let mut indef = ArrowRowBlock::new(d, k);
    indef.htt = array![[4.0_f64, 0.0, 0.0], [0.0, 1.0e-10, 0.0], [0.0, 0.0, -1.0],];
    indef.htbeta = array![[1.0_f64, 0.0], [0.0, 1.0], [0.5, 0.5]];
    indef.gt = array![0.0_f64, 0.0, 0.0];
    let gauge_e1 = array![0.0_f64, 1.0, 0.0];

    // Gauge deflation cannot manufacture a PD block: the −1.0 along e_2 is
    // genuine indefiniteness, not a near-null orbit, so deflating e_1 leaves
    // it negative and the closed-form deflation returns None.
    assert!(
        factor_gauge_deflated_evidence_row(&indef, d, std::slice::from_ref(&gauge_e1)).is_none(),
        "gauge deflation must NOT rescue a genuinely-indefinite non-gauge direction"
    );

    // Spectral deflation DISCOVERS the negative e_2 direction (and the flat
    // e_1) from the block's own eigendecomposition and stiffens BOTH to +1,
    // producing an SPD block. The two sub-floor eigenvalues (−1.0 and 1e-10
    // vs floor = 1e-8·4) are counted; the genuine e_0 (eigenvalue 4.0) is
    // preserved exactly.
    let spectral = factor_spectral_deflated_evidence_row(&indef, d)
        .expect("spectral deflation must condition the indefinite non-gauge block");
    assert_eq!(
        spectral.gauge_deflated_directions, 2,
        "the two sub-floor eigen-directions (−1.0 and 1e-10) must be unit-deflated"
    );
    // Reconstruct L Lᵀ: e_0 keeps 4.0; the two deflated axes each carry +1.
    let ls = &spectral.factor;
    let mut recon = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0_f64;
            for kk in 0..d {
                acc += ls[[i, kk]] * ls[[j, kk]];
            }
            recon[[i, j]] = acc;
        }
    }
    assert!(
        (recon[[0, 0]] - 4.0).abs() < 1.0e-9,
        "genuine direction e_0 must be preserved exactly; got {}",
        recon[[0, 0]]
    );
    assert!(
        (recon[[2, 2]] - 1.0).abs() < 1.0e-9,
        "the genuinely-indefinite direction e_2 must be deflated to unit \
             stiffness +1 (log 1 = 0, ρ-independent), NOT ridge-damped; got {}",
        recon[[2, 2]]
    );

    // The undamped evidence factor (tolerate_ill_conditioning, ridge_t = 0,
    // gauge passed in) now SUCCEEDS on this block via spectral deflation
    // rather than refusing — so the SAE driver gets a finite, BIAS-FREE
    // evidence cache and never falls back to a ρ-dependent ridge.
    let factored = factor_one_row_result(&indef, 0.0, d, 0, true, std::slice::from_ref(&gauge_e1))
        .expect("undamped evidence factor must condition the indefinite block by deflation");
    for a in 0..d {
        assert!(
            factored.factor[[a, a]].is_finite() && factored.factor[[a, a]] > 0.0,
            "deflated evidence factor must have a finite positive pivot at {a}; got {}",
            factored.factor[[a, a]]
        );
    }

    // Stationary block: the previously-indefinite e_2 direction is now
    // positive (genuine PD), the gauge direction e_1 stays near-null. The
    // undamped evidence factor must SUCCEED and return the EXACT Cholesky of
    // the block (with the unit-stiffness deflation on the gauge direction
    // contributing exactly +1 there, log(1) = 0 to the evidence) — NO ridge
    // bias. This is the converged state whose value/gradient must be
    // bit-identical to today's.
    let mut pd = ArrowRowBlock::new(d, k);
    pd.htt = array![[4.0_f64, 0.0, 0.0], [0.0, 1.0e-10, 0.0], [0.0, 0.0, 2.0],];
    pd.htbeta = indef.htbeta.clone();
    pd.gt = array![0.0_f64, 0.0, 0.0];

    let result = factor_one_row_result(&pd, 0.0, d, 0, true, std::slice::from_ref(&gauge_e1))
        .expect("undamped evidence factor must succeed on the genuinely-PD stationary block");
    // Exactly one gauge direction deflated; the non-gauge spectrum is
    // factored as-is (no ridge), so L Lᵀ reproduces H_tt on the two genuine
    // directions and the deflated gauge direction carries the +1 stiffness.
    assert_eq!(
        result.gauge_deflated_directions, 1,
        "exactly the single near-null gauge direction must be deflated"
    );
    let l = &result.factor;
    let mut reconstructed = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0_f64;
            for kk in 0..d {
                acc += l[[i, kk]] * l[[j, kk]];
            }
            reconstructed[[i, j]] = acc;
        }
    }
    // Genuine directions: exact, no ridge bias.
    assert!(
        (reconstructed[[0, 0]] - 4.0).abs() < 1.0e-12,
        "stationary factor must be the EXACT Cholesky on the genuine direction e_0; got {}",
        reconstructed[[0, 0]]
    );
    assert!(
        (reconstructed[[2, 2]] - 2.0).abs() < 1.0e-12,
        "stationary factor must be the EXACT Cholesky on the genuine direction e_2; got {}",
        reconstructed[[2, 2]]
    );
    // Gauge direction: raw curvature 1e-10 + unit Faddeev-Popov stiffness 1.0.
    assert!(
        (reconstructed[[1, 1]] - (1.0 + 1.0e-10)).abs() < 1.0e-9,
        "deflated gauge direction must carry exactly the +1 unit stiffness; got {}",
        reconstructed[[1, 1]]
    );
}

/// #1117 flicker guard: a per-row evidence block carrying ONE genuinely
/// indefinite direction (so spectral deflation runs) plus a small POSITIVE
/// eigenvalue parked right at the relative cutoff `floor = REL_FLOOR·max|λ|`
/// must report the SAME deflation count at two infinitesimally different
/// "ρ values" that straddle the bare floor. Without the hysteresis band the
/// positive near-floor eigenvalue would be counted as deflated on one side
/// (`λ ≤ floor`) and live on the other (`λ > floor`), flipping the per-row
/// count and tripping the quotient-dimension guard
/// (`record_evidence_gauge_deflation_count`) mid-optimization — the slow
/// seed/homotopy cascade. The genuine indefinite direction (the true
/// quotient null) is deflated on BOTH sides, so the count is stable.
#[test]
pub(crate) fn evidence_row_spectral_deflation_count_is_stable_across_the_cutoff() {
    let d = 3usize;
    let k = 1usize;
    // max|λ| = 4.0 ⇒ floor = SPECTRAL_DEFLATION_REL_FLOOR·4 = 4e-8. Place the
    // small positive eigenvalue just BELOW and just ABOVE the bare floor at
    // two ρ-walk iterates; the third direction is genuinely indefinite
    // (−1.0) so spectral deflation runs on both.
    let floor = SPECTRAL_DEFLATION_REL_FLOOR * 4.0;

    // The bare cutoff is the knife-edge: `λ ≤ floor` would deflate the lo
    // iterate and keep the hi iterate, flipping the count. The hysteresis
    // floor is `floor·(1−1e-2) = floor·0.99`, so picking both iterates
    // strictly ABOVE it (0.995·floor and 1.05·floor) keeps them on the same
    // (KEEP) side of the banded decision while still straddling the BARE
    // floor — exactly the flicker regime the fix removes.
    let near_floor_lo = floor * 0.995; // bare cutoff: deflated; banded: kept
    let near_floor_hi = floor * 1.05; // bare cutoff: live; banded: kept

    let mut block_lo = ArrowRowBlock::new(d, k);
    block_lo.htt = array![
        [4.0_f64, 0.0, 0.0],
        [0.0, near_floor_lo, 0.0],
        [0.0, 0.0, -1.0],
    ];
    block_lo.htbeta = array![[1.0_f64], [0.0], [0.5]];
    block_lo.gt = array![0.0_f64, 0.0, 0.0];

    let mut block_hi = block_lo.clone();
    block_hi.htt[[1, 1]] = near_floor_hi;

    let lo = factor_spectral_deflated_evidence_row(&block_lo, d)
        .expect("indefinite block must spectrally deflate (lo iterate)");
    let hi = factor_spectral_deflated_evidence_row(&block_hi, d)
        .expect("indefinite block must spectrally deflate (hi iterate)");

    // The genuine −1.0 quotient direction is deflated on both sides; the
    // small positive near-floor direction is KEPT on both sides thanks to
    // the hysteresis band, so the count does NOT flicker.
    assert_eq!(
        lo.gauge_deflated_directions, 1,
        "lo iterate: only the genuine indefinite direction is deflated"
    );
    assert_eq!(
        hi.gauge_deflated_directions, lo.gauge_deflated_directions,
        "deflation count must be STABLE across an eigenvalue straddling the \
             bare cutoff — the quotient-dimension guard must not trip mid-walk"
    );

    // Sanity: the bare (non-hysteresis) cutoff WOULD have split these two
    // iterates, confirming the test actually exercises the flicker regime.
    let bare_count = |lambda: f64| -> usize {
        let mut c = 0usize;
        for &l in &[4.0_f64, lambda, -1.0] {
            if !(l.is_finite() && l > floor) {
                c += 1;
            }
        }
        c
    };
    assert_ne!(
        bare_count(near_floor_lo),
        bare_count(near_floor_hi),
        "test must straddle the bare cutoff (else it proves nothing): the \
             un-banded decision flips the count, the banded one does not"
    );
}

/// #1118 (β-block analogue): a genuinely indefinite REDUCED SCHUR complement
/// — the state the OLMo K=8 capstone hits, where the per-row H_tt blocks are
/// deflated PD but the Schur subtraction drives a β-pivot negative (the
/// reported `-0.064 at index 256`) — must be conditioned by the evidence
/// dense factor through unit-stiffness spectral deflation rather than failing
/// the whole fit. The negative direction is stiffened to eigenvalue `+1`
/// (ρ-independent `log 1 = 0`), the genuine positive spectrum is preserved
/// exactly, and the result is PD so its Cholesky and `log|S|` are finite.
#[test]
pub(crate) fn evidence_dense_schur_deflates_indefinite_complement_at_unit_stiffness() {
    // A 3×3 symmetric Schur complement with one genuinely NEGATIVE eigenvalue
    // (−0.5 along e_1) and two healthy positive ones (4.0 along e_0, 2.0 along
    // e_2). The plain Cholesky must refuse it; the evidence deflation must
    // condition it to PD.
    let schur = array![[4.0_f64, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 2.0],];
    assert!(
        cholesky_lower(&schur).is_err(),
        "an indefinite Schur complement must be refused by the plain Cholesky"
    );

    let factor = factor_spectral_deflated_evidence_dense(&schur)
        .expect("indefinite Schur complement must spectrally deflate to a PD factor");

    // Reconstruct L Lᵀ and check the spectrum: genuine directions exact, the
    // deflated negative direction carries the +1 unit stiffness.
    let d = 3usize;
    let mut reconstructed = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0_f64;
            for kk in 0..d {
                acc += factor[[i, kk]] * factor[[j, kk]];
            }
            reconstructed[[i, j]] = acc;
        }
    }
    assert!(
        (reconstructed[[0, 0]] - 4.0).abs() < 1.0e-9,
        "genuine positive direction e_0 must be exact; got {}",
        reconstructed[[0, 0]]
    );
    assert!(
        (reconstructed[[2, 2]] - 2.0).abs() < 1.0e-9,
        "genuine positive direction e_2 must be exact; got {}",
        reconstructed[[2, 2]]
    );
    assert!(
        (reconstructed[[1, 1]] - 1.0).abs() < 1.0e-9,
        "deflated negative direction must carry exactly the +1 unit stiffness; got {}",
        reconstructed[[1, 1]]
    );
}

#[test]
pub(crate) fn sys_htbeta_materialize_row_sums_operator_and_dense_slab() {
    let mut sys = ArrowSchurSystem::new(1, 1, 3);
    sys.rows[0].htbeta = array![[0.25_f64, 0.5, 0.75]];
    sys.activate_dense_htbeta_supplement();
    sys.set_row_htbeta_operator(
        |row_idx, x, out| {
            assert_eq!(row_idx, 0);
            out[0] += 2.0 * x[0] - x[1] + 0.5 * x[2];
        },
        |row_idx, v, out| {
            assert_eq!(row_idx, 0);
            out[0] += 2.0 * v[0];
            out[1] -= v[0];
            out[2] += 0.5 * v[0];
        },
    );

    let htbeta = sys_htbeta_materialize_row(&sys, 0, &sys.rows[0]).unwrap();
    assert_eq!(htbeta, array![[2.25_f64, -0.5, 1.25]]);
}

/// Issue #195 / gam#578 / gam#845: when the per-row block is barely-PD at
/// `ridge_t = 0` (a rank-deficient atom), the per-row factor must
/// CONDITION it through the folded ridge escalation, and the full
/// `solve_with_lm_escalation_inner` must produce a finite Newton step
/// rather than aborting the whole fit.
///
/// Note (gam#845): per-row κ-conditioning bounds each block's inverse
/// spectrum, but it cannot on its own guarantee the *dense Schur
/// complement* `S = H_ββ − Σ_i H_tβᵀ(H_tt+ridge)⁻¹H_tβ` stays PD: the
/// per-row ceiling still admits a ~`1/κ_ceiling`-scale smallest pivot, so
/// `(H_tt+ridge)⁻¹` retains a ~`κ_ceiling`-scale eigenvalue that, after the
/// Schur subtraction, can drive `S` strongly indefinite when
/// `‖H_tβ‖²·κ_ceiling ≫ ‖H_ββ‖`. Outer LM ridge escalation is the correct,
/// principled recovery for that regime. The achievable invariant is
/// therefore: a finite, well-conditioned Newton step is produced (via a
/// bounded number of outer ridge escalations), NOT zero escalations.
#[test]
pub(crate) fn lm_escalation_recovers_from_ill_conditioned_row() {
    let n = 1;
    let d = 2;
    let k = 2;
    let mut sys = ArrowSchurSystem::new(n, d, k);
    // Same barely-PD row as the issue body.
    sys.rows[0].htt = array![[1.0_f64, 1.0], [1.0, 1.0 + 1e-14]];
    sys.rows[0].htbeta = array![[1.0_f64, 0.0], [0.0, 1.0]];
    sys.rows[0].gt = array![0.1_f64, -0.2];
    sys.hbb = array![[4.0_f64, 0.2], [0.2, 5.0]];
    sys.gb = array![0.3_f64, -0.1];

    // Direct factor at ridge_t=0 CONDITIONS the barely-PD block via the
    // folded per-row ridge escalation (gam#578: the advertised ridge
    // genuinely stabilizes the deficient direction instead of rejecting
    // it) and returns a well-conditioned factor satisfying the κ ceiling.
    let factor = factor_one_row(&sys.rows[0], 0.0, d, 0, false)
        .expect("barely-PD row must be conditioned, not rejected (gam#578)");
    let kappa = cholesky_factor_kappa_estimate(&factor);
    assert!(
        kappa.is_finite() && kappa <= safe_spd_kappa_max(d),
        "conditioned per-row factor must satisfy the κ ceiling; got κ={kappa:e}"
    );

    // The full LM-escalating wrapper produces a finite, well-conditioned
    // Newton step. Per-row conditioning alone cannot keep the dense Schur
    // complement PD here (κ_ceiling × ‖H_tβ‖² ≫ ‖H_ββ‖), so the proximal
    // wrapper escalates the outer ridge a bounded number of times — this
    // is the correct recovery (gam#845), not a failure.
    let options = ArrowSolveOptions::direct();
    let (delta_t, delta_beta, diag) = solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &options)
        .expect("LM escalation must recover from a barely-PD per-row block");
    for v in delta_t.iter().chain(delta_beta.iter()) {
        assert!(v.is_finite(), "recovered step must be finite: {v}");
    }
    assert!(
        diag.ridge_escalations <= DEFAULT_PROXIMAL_MAX_ATTEMPTS,
        "recovery must use a bounded number of outer ridge escalations; got {}",
        diag.ridge_escalations
    );
}

/// `latent_block_inverse_diagonal` must reproduce the `t`-block diagonal of
/// the dense bordered-arrow inverse `(H⁻¹)_tt` to machine precision.
///
/// Build a small `(N=3, d=2, K=2)` arrow system, factor it through the
/// real solve to obtain an [`ArrowFactorCache`], then assemble the full
/// dense `(N·d + K) × (N·d + K)` Hessian from the same per-row blocks,
/// invert it via dense Cholesky, and compare diagonals.
#[test]
pub(crate) fn latent_block_inverse_diagonal_matches_dense() {
    let n = 3usize;
    let d = 2usize;
    let k = 2usize;
    let mut sys = ArrowSchurSystem::new(n, d, k);

    // Distinct, well-conditioned per-row blocks and cross-blocks.
    sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
    sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
    sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
    sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
    sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
    sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
    for row in sys.rows.iter_mut() {
        row.gt = array![0.0_f64, 0.0];
    }
    // SPD shared block; the full bordered H must stay PD.
    sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
    sys.gb = array![0.0_f64, 0.0];

    let options = ArrowSolveOptions::direct();
    let (_delta_t, _delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("direct arrow solve should factor this SPD system");

    // Assemble the dense bordered-arrow Hessian H (t-coords first, then β).
    let dim = n * d + k;
    let mut h = Array2::<f64>::zeros((dim, dim));
    for i in 0..n {
        let base = i * d;
        // H_tt^(i) block.
        for r in 0..d {
            for c in 0..d {
                h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
            }
        }
        // H_tβ^(i) (d×K) and its transpose into the β border.
        for r in 0..d {
            for c in 0..k {
                let v = sys.rows[i].htbeta[[r, c]];
                h[[base + r, n * d + c]] = v;
                h[[n * d + c, base + r]] = v;
            }
        }
    }
    // H_ββ.
    for r in 0..k {
        for c in 0..k {
            h[[n * d + r, n * d + c]] = sys.hbb[[r, c]];
        }
    }

    // Dense inverse via Cholesky against the identity.
    let l = cholesky_lower(&h).expect("assembled bordered H must be SPD");
    let h_inv = cholesky_solve_matrix(&l, &Array2::<f64>::eye(dim));

    let diag = cache
        .latent_block_inverse_diagonal()
        .expect("dense Schur cache must support the selected-inverse diagonal");
    assert_eq!(diag.len(), n * d);
    for i in 0..n {
        for j in 0..d {
            let idx = i * d + j; // homogeneous system ⇒ row_offsets[i] == i*d.
            let expected = h_inv[[idx, idx]];
            let got = diag[idx];
            assert!(
                (got - expected).abs() < 1e-9,
                "row {i} axis {j}: selected-inverse diag {got} vs dense {expected}"
            );
        }
    }

    // The per-(atom, axis) trace is a sum over the relevant indices; e.g.
    // tr[(H⁻¹)_tt] over all latent coords equals the dense t-block trace.
    let trace_selected: f64 = diag.iter().sum();
    let trace_dense: f64 = (0..n * d).map(|idx| h_inv[[idx, idx]]).sum();
    assert!(
        (trace_selected - trace_dense).abs() < 1e-9,
        "full latent trace {trace_selected} vs dense {trace_dense}"
    );
}

/// `full_inverse_apply` (#1006 IFT/adjoint back-solve) must reproduce the dense
/// bordered-arrow inverse applied to an arbitrary arrow-layout RHS, and
/// solving against the system's own gradient must reproduce the Newton
/// step the solver itself returned (`Δ = H⁻¹g`) — both to near machine
/// precision on the ridge-0 Direct factor.
#[test]
pub(crate) fn full_inverse_apply_matches_dense_inverse_and_newton_step() {
    let n = 3usize;
    let d = 2usize;
    let k = 2usize;
    let mut sys = ArrowSchurSystem::new(n, d, k);
    sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
    sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
    sys.rows[0].gt = array![0.4_f64, -0.7];
    sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
    sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
    sys.rows[1].gt = array![-0.2_f64, 0.9];
    sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
    sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
    sys.rows[2].gt = array![1.1_f64, 0.3];
    sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
    sys.gb = array![0.5_f64, -0.8];

    let options = ArrowSolveOptions::direct();
    let (delta_t, delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("direct arrow solve should factor this SPD system");

    // (a) The solver returns the DESCENT step Δ = −H⁻¹g; full_inverse_apply is the
    // bare inverse application H⁻¹g, so u must equal −Δ exactly.
    let mut g_t = Array1::<f64>::zeros(n * d);
    for i in 0..n {
        for j in 0..d {
            g_t[i * d + j] = sys.rows[i].gt[j];
        }
    }
    let (u_t, u_beta) = cache
        .full_inverse_apply(g_t.view(), sys.gb.view())
        .expect("full_inverse_apply on the ridge-0 Direct cache");
    for idx in 0..n * d {
        assert!(
            (u_t[idx] + delta_t[idx]).abs() < 1e-10,
            "t[{idx}]: full_inverse_apply {} vs −(Newton step) {}",
            u_t[idx],
            -delta_t[idx]
        );
    }
    for c in 0..k {
        assert!(
            (u_beta[c] + delta_beta[c]).abs() < 1e-10,
            "beta[{c}]: full_inverse_apply {} vs −(Newton step) {}",
            u_beta[c],
            -delta_beta[c]
        );
    }

    // (b) Arbitrary RHS vs the dense bordered inverse.
    let dim = n * d + k;
    let mut h = Array2::<f64>::zeros((dim, dim));
    for i in 0..n {
        let base = i * d;
        for r in 0..d {
            for c in 0..d {
                h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
            }
            for c in 0..k {
                let v = sys.rows[i].htbeta[[r, c]];
                h[[base + r, n * d + c]] = v;
                h[[n * d + c, base + r]] = v;
            }
        }
    }
    for r in 0..k {
        for c in 0..k {
            h[[n * d + r, n * d + c]] = sys.hbb[[r, c]];
        }
    }
    let l = cholesky_lower(&h).expect("assembled bordered H must be SPD");
    let mut w_full = Array1::<f64>::zeros(dim);
    for (idx, v) in w_full.iter_mut().enumerate() {
        *v = 0.3 + 0.17 * (idx as f64) * (if idx % 2 == 0 { 1.0 } else { -1.0 });
    }
    let dense_u = cholesky_solve_vector(&l, &w_full);
    let (u_t2, u_beta2) = cache
        .full_inverse_apply(
            w_full.slice(ndarray::s![..n * d]),
            w_full.slice(ndarray::s![n * d..]),
        )
        .expect("full_inverse_apply on arbitrary RHS");
    for idx in 0..n * d {
        assert!(
            (u_t2[idx] - dense_u[idx]).abs() < 1e-10,
            "t[{idx}]: full_inverse_apply {} vs dense {}",
            u_t2[idx],
            dense_u[idx]
        );
    }
    for c in 0..k {
        assert!(
            (u_beta2[c] - dense_u[n * d + c]).abs() < 1e-10,
            "beta[{c}]: full_inverse_apply {} vs dense {}",
            u_beta2[c],
            dense_u[n * d + c]
        );
    }
}

/// `schur_inverse_apply` / `schur_inverse_block` must reproduce the
/// β-block of the dense bordered-arrow inverse `(H⁻¹)_ββ = S_β⁻¹`, and a
/// caller-assembled `tr(S_β⁻¹ M)` must match the dense Kron-block trace —
/// the β-side analogue used by the SAE λ_smooth Fellner-Schall step.
#[test]
pub(crate) fn schur_inverse_beta_block_matches_dense() {
    let n = 3usize;
    let d = 2usize;
    let k = 2usize;
    let mut sys = ArrowSchurSystem::new(n, d, k);
    sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
    sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
    sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
    sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
    sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
    sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
    for row in sys.rows.iter_mut() {
        row.gt = array![0.0_f64, 0.0];
    }
    sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
    sys.gb = array![0.0_f64, 0.0];

    let options = ArrowSolveOptions::direct();
    let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
        .expect("direct arrow solve should factor this SPD system");

    // Dense bordered H and its inverse (same assembly as the t-block test).
    let dim = n * d + k;
    let mut h = Array2::<f64>::zeros((dim, dim));
    for i in 0..n {
        let base = i * d;
        for r in 0..d {
            for c in 0..d {
                h[[base + r, base + c]] = sys.rows[i].htt[[r, c]];
            }
        }
        for r in 0..d {
            for c in 0..k {
                let v = sys.rows[i].htbeta[[r, c]];
                h[[base + r, n * d + c]] = v;
                h[[n * d + c, base + r]] = v;
            }
        }
    }
    for r in 0..k {
        for c in 0..k {
            h[[n * d + r, n * d + c]] = sys.hbb[[r, c]];
        }
    }
    let l = cholesky_lower(&h).expect("assembled bordered H must be SPD");
    let h_inv = cholesky_solve_matrix(&l, &Array2::<f64>::eye(dim));

    // The β-block of H⁻¹ is the bottom-right K×K corner.
    let beta_off = n * d;

    // schur_inverse_apply against each unit column reproduces the full
    // β-block (every entry, not just the diagonal).
    for col in 0..k {
        let mut e = Array1::<f64>::zeros(k);
        e[col] = 1.0;
        let x = cache
            .schur_inverse_apply(e.view())
            .expect("dense Schur cache must support schur_inverse_apply");
        for r in 0..k {
            let expected = h_inv[[beta_off + r, beta_off + col]];
            assert!(
                (x[r] - expected).abs() < 1e-9,
                "S_β⁻¹[{r},{col}] {} vs dense {expected}",
                x[r]
            );
        }
    }

    // Caller-assembled Kron trace tr(S_β⁻¹ M) for a single atom block
    // M = A_k ⊗ I_p with K = M_k · p. Here M_k = 1, p = 2 ⇒ K = 2, so
    // A_k is 1×1 = [a] and M = a·I_2. tr(S_β⁻¹ M) = a·tr(S_β⁻¹).
    let a_scalar = 0.75_f64;
    let mut trace = 0.0_f64;
    for col in 0..k {
        // (A_k ⊗ I_p) e_col = a_scalar · e_col for this M_k=1 block.
        let mut m_col = Array1::<f64>::zeros(k);
        m_col[col] = a_scalar;
        let z = cache
            .schur_inverse_apply(m_col.view())
            .expect("schur_inverse_apply");
        trace += z[col];
    }
    let trace_dense: f64 = a_scalar
        * (0..k)
            .map(|j| h_inv[[beta_off + j, beta_off + j]])
            .sum::<f64>();
    assert!(
        (trace - trace_dense).abs() < 1e-9,
        "Kron-block trace {trace} vs dense {trace_dense}"
    );

    // schur_inverse_block must reproduce a contiguous dense sub-block of
    // (H⁻¹)_ββ — both the full β-block and an interior single-coordinate
    // window — and be exactly symmetric.
    let full = cache
        .schur_inverse_block(0..k)
        .expect("dense Schur cache must support schur_inverse_block");
    assert_eq!(full.dim(), (k, k));
    for r in 0..k {
        for c in 0..k {
            let expected = h_inv[[beta_off + r, beta_off + c]];
            assert!(
                (full[[r, c]] - expected).abs() < 1e-9,
                "block[{r},{c}] {} vs dense {expected}",
                full[[r, c]]
            );
            assert!(
                (full[[r, c]] - full[[c, r]]).abs() < 1e-12,
                "schur_inverse_block must be symmetric at [{r},{c}]"
            );
        }
    }
    let sub = cache
        .schur_inverse_block(1..k)
        .expect("interior block must be supported");
    assert_eq!(sub.dim(), (k - 1, k - 1));
    assert!(
        (sub[[0, 0]] - h_inv[[beta_off + 1, beta_off + 1]]).abs() < 1e-9,
        "interior block [1,1] {} vs dense {}",
        sub[[0, 0]],
        h_inv[[beta_off + 1, beta_off + 1]]
    );
    // Out-of-range block must error rather than panic.
    assert!(cache.schur_inverse_block(0..(k + 1)).is_err());
}

/// Evidence/log-det mode: a per-row `H_tt` that is PD but ill-conditioned
/// (κ above the safe-Schur ceiling) is handled differently by the two
/// solve paths. The default `direct()` path conditions each row to the
/// safe-Schur κ ceiling; when that per-row conditioning is insufficient to
/// keep the *dense Schur complement* PD (gam#845), the single-shot solve
/// correctly reports a recoverable factorization error and the
/// LM-escalating wrapper recovers it with a finite, well-conditioned step.
///
/// `with_ill_conditioning_tolerated()` accepts the RAW (undamped) blocks.
/// Its contract has two sides, pinned on two fixtures:
///   * row-PD but assembled-INDEFINITE H (strong coupling into near-null
///     t-directions) → honest refusal. Per-row PD does not imply bordered-
///     system PD, and an exact `log|H|` does not exist on the Cholesky
///     branch — fabricating one would corrupt the evidence.
///   * row κ ≈ 1e9 but assembled H genuinely PD (coupling subordinate to
///     the weak curvature) → a usable cache whose log-determinant equals
///     the exact dense `log|H|`, undistorted by any κ-ceiling ridge. This
///     is the SAE evidence path under a wide ARD α sweep.
#[test]
pub(crate) fn ill_conditioning_tolerated_returns_cache_with_exact_logdet() {
    let n = 2usize;
    let d = 2usize;
    let k = 2usize;
    let mut sys = ArrowSchurSystem::new(n, d, k);
    // Barely-PD rows: second pivot ~1e-9 of the first ⇒ κ ≈ 1e9, above
    // the safe-Schur ceiling but genuinely PD (Cholesky succeeds).
    sys.rows[0].htt = array![[1.0_f64, 0.0], [0.0, 1e-9]];
    sys.rows[0].htbeta = array![[0.3_f64, 0.1], [0.05, 0.2]];
    sys.rows[1].htt = array![[2.0_f64, 0.0], [0.0, 2e-9]];
    sys.rows[1].htbeta = array![[0.2_f64, -0.1], [0.1, 0.15]];
    for row in sys.rows.iter_mut() {
        row.gt = array![0.0_f64, 0.0];
    }
    sys.hbb = array![[5.0_f64, 0.3], [0.3, 4.0]];
    sys.gb = array![0.0_f64, 0.0];

    // factor_one_row conditions each barely-PD per-row block to the
    // safe-Schur κ ceiling (gam#578): the raw block fails the ceiling but
    // the ridge-lifted factor satisfies it. Verify the per-row contract
    // directly — this is what per-row conditioning genuinely guarantees.
    for i in 0..n {
        let factor = factor_one_row(&sys.rows[i], 0.0, d, i, false)
            .expect("barely-PD row must be conditioned, not rejected (gam#578)");
        let kappa = cholesky_factor_kappa_estimate(&factor);
        assert!(
            kappa.is_finite() && kappa <= safe_spd_kappa_max(d),
            "conditioned per-row factor {i} must satisfy the safe-Schur κ ceiling; got κ={kappa:e}"
        );
    }

    // Per-row conditioning alone cannot keep the dense Schur complement PD
    // for these inputs (κ_ceiling × ‖H_tβ‖² ≫ ‖H_ββ‖, gam#845), so the
    // single-shot strict solve reports a recoverable factorization error
    // rather than a finite step.
    let single_shot =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &ArrowSolveOptions::direct());
    assert!(
        matches!(
            single_shot,
            Err(ArrowSchurError::SchurFactorFailed { .. })
                | Err(ArrowSchurError::PerRowFactorIllConditioned { .. })
                | Err(ArrowSchurError::PcgFailed { .. })
        ),
        "single-shot strict direct() cannot keep the dense Schur PD with per-row \
             conditioning alone; expected a recoverable factorization error, got {single_shot:?}"
    );

    // The LM-escalating wrapper is the correct recovery: a bounded number
    // of outer ridge escalations yields a finite, well-conditioned step.
    let (strict_dt, strict_db, strict_diag) =
        solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &ArrowSolveOptions::direct())
            .expect("LM escalation must recover the ill-conditioned strict solve (gam#845)");
    for v in strict_dt.iter().chain(strict_db.iter()) {
        assert!(v.is_finite(), "recovered strict step must be finite: {v}");
    }
    assert!(
        strict_diag.ridge_escalations <= DEFAULT_PROXIMAL_MAX_ATTEMPTS,
        "recovery must use a bounded number of outer ridge escalations; got {}",
        strict_diag.ridge_escalations
    );

    // Evidence mode accepts the RAW (undamped) blocks. For THIS system the
    // honest answer is refusal: each per-row `H_tt` is PD in isolation, but
    // the strong coupling into the near-null t-directions makes the
    // assembled bordered H indefinite (its true Schur complement has a
    // ≈ −7.5e6 leading pivot; the full spectrum has two negative
    // eigenvalues). An exact log|H| does not exist on the Cholesky branch,
    // and tolerating ill-CONDITIONING must never fabricate a determinant
    // for an in-DEFINITE system — the SchurFactorFailed refusal is the
    // contract, not a defect.
    let opts = ArrowSolveOptions::direct().with_ill_conditioning_tolerated();
    let tolerate_indefinite = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &opts);
    assert!(
        matches!(
            tolerate_indefinite,
            Err(ArrowSchurError::SchurFactorFailed { .. })
        ),
        "tolerate mode must refuse the indefinite assembled H rather than fabricate \
             a log-determinant; got {tolerate_indefinite:?}"
    );

    // The regime the tolerate flag exists for: per-row κ ≈ 1e9 (above the
    // safe-Schur ceiling, so the strict path would ridge-condition the row
    // and distort the determinant) yet the assembled H is genuinely PD
    // because the coupling into the near-null t-directions is subordinate
    // to their curvature (‖H_tβ row‖² ≲ λ_min(H_tt)·λ_min(H_ββ)). Evidence
    // mode must factor the RAW blocks and report the EXACT dense log|H|,
    // undistorted by any κ-ceiling ridge.
    let mut pd_sys = ArrowSchurSystem::new(n, d, k);
    pd_sys.rows[0].htt = array![[1.0_f64, 0.0], [0.0, 1e-9]];
    pd_sys.rows[0].htbeta = array![[0.3_f64, 0.1], [3e-6, 1e-6]];
    pd_sys.rows[1].htt = array![[2.0_f64, 0.0], [0.0, 2e-9]];
    pd_sys.rows[1].htbeta = array![[0.2_f64, -0.1], [2e-6, 4e-6]];
    for row in pd_sys.rows.iter_mut() {
        row.gt = array![0.0_f64, 0.0];
    }
    pd_sys.hbb = array![[5.0_f64, 0.3], [0.3, 4.0]];
    pd_sys.gb = array![0.0_f64, 0.0];

    let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&pd_sys, 0.0, 0.0, &opts)
        .expect("tolerate mode must factor the ill-conditioned-but-PD system");

    // Cache log-determinant (Σ log|H_tt^i| + log|S_β|) must equal the exact
    // dense log|H|, regardless of conditioning — the whole point.
    let (log_det_tt, log_det_schur) = cache.arrow_log_det();
    let log_det_cache = log_det_tt + log_det_schur.expect("dense Schur factor present");

    // Dense reference: assemble H and take log|H| = 2 Σ log L_ii.
    let dim = n * d + k;
    let mut h = Array2::<f64>::zeros((dim, dim));
    for i in 0..n {
        let base = i * d;
        for r in 0..d {
            for c in 0..d {
                h[[base + r, base + c]] = pd_sys.rows[i].htt[[r, c]];
            }
        }
        for r in 0..d {
            for c in 0..k {
                let v = pd_sys.rows[i].htbeta[[r, c]];
                h[[base + r, n * d + c]] = v;
                h[[n * d + c, base + r]] = v;
            }
        }
    }
    for r in 0..k {
        for c in 0..k {
            h[[n * d + r, n * d + c]] = pd_sys.hbb[[r, c]];
        }
    }
    let lh = cholesky_lower(&h).expect("assembled bordered H must be SPD");
    let log_det_dense: f64 = 2.0 * (0..dim).map(|i| lh[[i, i]].ln()).sum::<f64>();

    assert!(
        (log_det_cache - log_det_dense).abs() < 1e-6,
        "tolerated-cache log|H| {log_det_cache} vs dense {log_det_dense}"
    );

    // Selected-inverse traces must still be available from the cache.
    let tdiag = cache
        .latent_block_inverse_diagonal()
        .expect("tolerated cache must support latent_block_inverse_diagonal");
    assert_eq!(tdiag.len(), n * d);
    assert!(tdiag.iter().all(|v| v.is_finite()));
}

#[test]
pub(crate) fn arrow_factor_slab_accessor_matches_array_blocks_bitwise() {
    let blocks = vec![
        array![[1.0_f64]],
        array![[2.0_f64, 0.0], [0.25, 3.0]],
        array![[4.0_f64, 0.0, 0.0], [0.5, 5.0, 0.0], [-0.25, 0.75, 6.0]],
    ];
    let slab = ArrowFactorSlab::from_blocks(blocks.clone());
    assert_eq!(slab.len(), blocks.len());
    for row in 0..blocks.len() {
        let view = slab.factor(row);
        assert_eq!(view.dim(), blocks[row].dim());
        for r in 0..blocks[row].nrows() {
            for c in 0..blocks[row].ncols() {
                assert_eq!(view[[r, c]].to_bits(), blocks[row][[r, c]].to_bits());
            }
        }
    }
}

pub(crate) fn fixed_row_kernel_fixture<const D: usize>() -> (ArrowRowBlock, Array1<f64>) {
    let mut row = ArrowRowBlock::new(D, 0);
    for r in 0..D {
        for c in 0..D {
            row.htt[[r, c]] = if r == c {
                4.0 + r as f64
            } else {
                0.03125 * ((r + c + 1) as f64)
            };
        }
    }
    let rhs = Array1::from_iter((0..D).map(|i| 0.5 + i as f64 * 0.25));
    (row, rhs)
}

pub(crate) fn assert_fixed_row_kernels_match_dynamic<const D: usize>() -> usize {
    let (row, rhs) = fixed_row_kernel_fixture::<D>();
    let ridge = 0.125_f64;
    let fixed = factor_row_block_cholesky_fixed::<D>(&row, ridge).expect("fixed factor");
    let dynamic = factor_row_block_cholesky_dynamic(&row, ridge, D).expect("dynamic factor");
    for r in 0..D {
        for c in 0..D {
            assert_eq!(
                fixed[[r, c]].to_bits(),
                dynamic[[r, c]].to_bits(),
                "factor mismatch at D={D} ({r},{c})"
            );
        }
    }

    let fixed_solve = cholesky_solve_vector_fixed::<D>(fixed.view(), rhs.view());
    let dynamic_solve = cholesky_solve_vector(dynamic.view(), rhs.view());
    for i in 0..D {
        assert_eq!(
            fixed_solve[i].to_bits(),
            dynamic_solve[i].to_bits(),
            "solve mismatch at D={D} index {i}"
        );
    }
    D
}

#[test]
pub(crate) fn fixed_row_kernels_match_dynamic_path_bitwise() {
    let checked = assert_fixed_row_kernels_match_dynamic::<1>()
        + assert_fixed_row_kernels_match_dynamic::<2>()
        + assert_fixed_row_kernels_match_dynamic::<3>()
        + assert_fixed_row_kernels_match_dynamic::<4>();
    assert_eq!(checked, 10);
}

/// Build a small, well-conditioned dense Direct arrow system: `n` rows of
/// `d×d` PD blocks, small `d×k` cross blocks, a diagonally-dominant `k×k`
/// border. Used to exercise the #1017 production device-routing seam on the
/// host (where the device declines, so the CPU path must answer unchanged).
pub(crate) fn dense_direct_system(n: usize, d: usize, k: usize) -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(n, d, k);
    for (i, row) in sys.rows.iter_mut().enumerate() {
        for r in 0..d {
            for c in 0..d {
                row.htt[[r, c]] = if r == c { 4.0 + (i % 3) as f64 } else { 0.1 };
            }
            row.gt[r] = 0.05 * ((i + r + 1) as f64).sin();
            for c in 0..k {
                row.htbeta[[r, c]] = 0.01 * (((i + 1) * (c + 1)) as f64).cos();
            }
        }
    }
    for r in 0..k {
        sys.gb[r] = 0.02 * ((r + 1) as f64).cos();
        for c in 0..k {
            sys.hbb[[r, c]] = if r == c { 6.0 } else { 0.0 };
        }
    }
    sys.refresh_row_hessian_fingerprint();
    sys
}

/// The #1017 work-based dispatch predicate must admit LLM/SAE shapes (few
/// rows, wide border) and reject tiny shapes where launch latency wins.
#[test]
pub(crate) fn device_dispatch_predicate_gates_on_work_not_rows() {
    let policy = crate::gpu::policy::GpuDispatchPolicy::default();
    // Tiny: below the DEVICE_LOOP_MIN_P border floor → never on device.
    assert!(!policy.dense_hessian_work_target_is_gpu(300, 8));
    // LLM/SAE: 2000 rows × a few-thousand-wide border clears both the
    // min-p floor and the 2·n·p² flop threshold.
    assert!(policy.dense_hessian_work_target_is_gpu(2_000, 4_096));
}

/// #1017 Phase-1 call-site re-key: the live matvec-injection gate
/// (`maybe_inject_gpu_schur_matvec`) now keys on the CG-amortised
/// `reduced_schur_matvec_should_offload(rows, k, sys.d, cg_iters)` predicate
/// rather than the dense-Direct `(rows, k)` floor. This asserts the predicate
/// the gate consults — with the exact `cg_iters` the gate derives from the
/// options (`pcg.max_iterations.min(trust_region.max_iterations)`) — fires for
/// the SAE LLM shape (n~2000 rows × k~2048 border × d~8 frame depth) while
/// staying off for tiny shapes where launch latency dominates. The gate's
/// device-presence short-circuit (`GpuRuntime::global()?`) makes the helper
/// itself return `None` on a CPU-only host, so the routing logic is asserted
/// through the predicate it consults (the device==CPU 1e-10 numeric parity is
/// asserted by the box harness).
#[test]
pub(crate) fn matvec_gate_engages_for_llm_shape_off_for_tiny() {
    let policy = crate::gpu::policy::GpuDispatchPolicy::default();
    // The cg_iters the live gate derives from default options is exactly the
    // budget the PCG loop launches with.
    let options = ArrowSolveOptions::inexact_pcg();
    let cg_iters = options
        .pcg
        .max_iterations
        .min(options.trust_region.max_iterations);
    assert!(cg_iters > 0);

    // SAE LLM shape: few row blocks, wide border, modest frame depth. The
    // dense-Direct `(rows, k)` floor that the gate used to consult ignores the
    // frame depth `d` and the CG amortisation — assert the NEW predicate the
    // re-keyed gate consults admits it.
    let (n_llm, k_llm, d_llm) = (2_000_usize, 2_048_usize, 8_usize);
    assert!(policy.reduced_schur_matvec_should_offload(n_llm, k_llm, d_llm, cg_iters));

    // Tiny shape: narrow border below the device-loop floor → the gate stays
    // off regardless of the CG budget (launch latency dominates).
    assert!(!policy.reduced_schur_matvec_should_offload(30, 8, 2, cg_iters));
    // CPU-canary `(300, 8)` shape from the dense floor's own tests: still off.
    assert!(!policy.reduced_schur_matvec_should_offload(300, 8, 4, cg_iters));
}

/// On a host without a CUDA device the production seam must decline (return
/// `None`), so `solve_arrow_newton_step_core` runs the unchanged CPU path
/// and the result equals the direct CPU artifacts solve bit-for-bit.
#[test]
pub(crate) fn device_seam_declines_without_gpu_and_matches_cpu() {
    if crate::gpu::runtime::GpuRuntime::global().is_some() {
        // On a CUDA host the device may legitimately serve the step; this
        // host-only invariant does not apply. The box harness asserts the
        // device==CPU 1e-10 parity instead.
        return;
    }
    let sys = dense_direct_system(6, 2, 4);
    let options = ArrowSolveOptions::direct();

    // The seam helpers both decline when no device is present.
    assert!(try_device_arrow_direct(&sys, 0.0, 0.0, &options).is_none());
    assert!(maybe_inject_gpu_schur_matvec(&sys, 0.0, 0.0, &options).is_none());

    // The public core entry therefore equals the direct CPU artifacts solve.
    let (dt_core, db_core, diag) =
        solve_arrow_newton_step_core(&sys, 0.0, 0.0, &options).expect("core solve");
    assert!(
        !diag.used_device_arrow,
        "no device present, so the solve must not be flagged device-served"
    );
    let artifacts =
        solve_arrow_newton_step_artifacts(&sys, 0.0, 0.0, &options).expect("artifacts solve");
    for (a, b) in dt_core.iter().zip(artifacts.delta_t.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "Δt must be bit-identical to CPU");
    }
    for (a, b) in db_core.iter().zip(artifacts.delta_beta.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "Δβ must be bit-identical to CPU");
    }
}

/// #1014: the streaming reduced solve under certified mixed precision must
/// agree with the f64 solve to the backward-error certificate, and — the
/// load-bearing invariant — the evidence log-determinant must be UNCHANGED
/// (bit-for-bit) because it is read from the f64 reduced-Schur factor, never
/// the f32 solve.
#[test]
pub(crate) fn streaming_mixed_precision_matches_f64_and_keeps_logdet_f64() {
    let sys = dense_direct_system(40, 3, 6);

    let f64_options = ArrowSolveOptions::direct().with_streaming_chunk_size(Some(8));
    let mp_options = f64_options
        .clone()
        .with_mixed_precision_policy(MixedPrecisionPolicy::certified());
    assert!(matches!(
        f64_options.mixed_precision,
        MixedPrecisionPolicy::Off
    ));

    let mut s_f64 = StreamingArrowSchur::from_system(&sys, 8);
    let (_, db_f64, _) = s_f64
        .solve(0.0, 0.0, &f64_options)
        .expect("f64 streaming solve");
    let mut s_mp = StreamingArrowSchur::from_system(&sys, 8);
    let (_, db_mp, _) = s_mp
        .solve(0.0, 0.0, &mp_options)
        .expect("mp streaming solve");

    // The mixed-precision Δβ matches the f64 Δβ to the certified tolerance.
    let mut max_abs = 0.0_f64;
    for (a, b) in db_f64.iter().zip(db_mp.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 1e-7,
        "mixed-precision Δβ deviates from f64 by {max_abs:e}, above the certified tolerance"
    );

    // Evidence log-determinant: f64 regardless of the Δβ precision policy.
    let mut ld_f64 = StreamingArrowSchur::from_system(&sys, 8);
    let logdet_f64 = ld_f64
        .exact_arrow_log_det(0.0, 0.0, &f64_options)
        .expect("f64 logdet");
    let mut ld_mp = StreamingArrowSchur::from_system(&sys, 8);
    let logdet_mp = ld_mp
        .exact_arrow_log_det(0.0, 0.0, &mp_options)
        .expect("mp logdet");
    assert_eq!(
        logdet_f64.to_bits(),
        logdet_mp.to_bits(),
        "evidence log|H| must stay bit-for-bit f64 under the mixed-precision policy"
    );
}

/// The streaming dispatch turns mixed precision ON by default (#1014) but
/// honors an explicit caller policy.
#[test]
pub(crate) fn streaming_mixed_precision_default_upgrades_only_off() {
    let off = ArrowSolveOptions::direct();
    assert!(matches!(
        off.with_streaming_mixed_precision_default().mixed_precision,
        MixedPrecisionPolicy::Certified { .. }
    ));
    let pinned = ArrowSolveOptions::direct().with_mixed_precision_policy(MixedPrecisionPolicy::Off);
    // An explicit Off is still upgraded (it is the inherited default), but a
    // caller that pinned Certified keeps its own parameters.
    let custom =
        ArrowSolveOptions::direct().with_mixed_precision_policy(MixedPrecisionPolicy::Certified {
            max_refinement_steps: 1,
            residual_relative_tolerance: 1e-6,
            kappa_unit_roundoff_margin: 0.25,
        });
    match custom
        .with_streaming_mixed_precision_default()
        .mixed_precision
    {
        MixedPrecisionPolicy::Certified {
            max_refinement_steps,
            ..
        } => assert_eq!(max_refinement_steps, 1, "explicit policy preserved"),
        MixedPrecisionPolicy::Off => panic!("explicit Certified must not be downgraded"),
    }
    // `pinned` documents that Off is the upgrade trigger.
    assert!(matches!(pinned.mixed_precision, MixedPrecisionPolicy::Off));
}

// ----------------------------------------------------------------------
// #1038 cross-row IBP Woodbury: value + log-determinant + adjoint must all
// describe the SAME dense `H_full = H₀' + U D Uᵀ`. These checks build the
// dense bordered `H_full` explicitly (the i≠j cross-row terms layered onto
// the assembled self-term `H₀`) and assert the cache reproduces its
// log-determinant, its full inverse, its latent inverse diagonal, and the
// Newton step `H_full⁻¹(−g)` exactly.
// ----------------------------------------------------------------------

/// Build a small `(N, d, K_beta)` system with `R` IBP atom columns whose
/// logit slots are the first `R` latent coords of every row. Returns the
/// system (with the self term `d_k·z'_ik²` already on the logit diagonals,
/// as the assembly writes it), the source, and the per-(row, atom) `z'_ik`.
pub(crate) fn build_ibp_woodbury_fixture() -> (ArrowSchurSystem, IbpCrossRowSource, Vec<Vec<f64>>) {
    let n = 3usize;
    let d = 2usize;
    let k_beta = 2usize;
    let r = 2usize; // two atom columns, supported on logit slots 0 and 1.
    let mut sys = ArrowSchurSystem::new(n, d, k_beta);
    // Base (no-self) per-row latent blocks + cross-blocks + gradient.
    sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
    sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
    sys.rows[0].gt = array![0.4_f64, -0.7];
    sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
    sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
    sys.rows[1].gt = array![-0.2_f64, 0.9];
    sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
    sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
    sys.rows[2].gt = array![1.1_f64, 0.3];
    sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
    sys.gb = array![0.5_f64, -0.8];

    // IBP source: d_k coefficients (one positive, one negative — exercise the
    // indefinite-capacitance LU path) and z'_ik per (row, atom).
    let d_coef = array![0.6_f64, -0.35];
    let zprime = vec![
        vec![0.9_f64, 0.4], // row 0: z'_00, z'_01
        vec![-0.5, 0.8],    // row 1
        vec![0.7, -0.6],    // row 2
    ];
    let mut entries = Vec::new();
    for i in 0..n {
        for k in 0..r {
            // logit slot for atom k is latent coord k of row i.
            let g = sys.row_offsets[i] + k;
            entries.push((g, k, zprime[i][k]));
        }
    }
    // Write the self term d_k·z'_ik² onto the logit diagonals (as assembly does).
    for i in 0..n {
        for k in 0..r {
            sys.rows[i].htt[[k, k]] += d_coef[k] * zprime[i][k] * zprime[i][k];
        }
    }
    let source = IbpCrossRowSource {
        r,
        d: d_coef,
        entries,
    };
    (sys, source, zprime)
}

/// Assemble the dense bordered `H_full` (with the i≠j cross-row terms) from
/// the self-term system + source.
pub(crate) fn dense_h_full(
    sys: &ArrowSchurSystem,
    source: &IbpCrossRowSource,
    zprime: &[Vec<f64>],
) -> Array2<f64> {
    let n = sys.rows.len();
    let d = sys.d;
    let k_beta = sys.k;
    let dim = n * d + k_beta;
    let mut h = Array2::<f64>::zeros((dim, dim));
    for i in 0..n {
        let base = i * d;
        for rr in 0..d {
            for cc in 0..d {
                h[[base + rr, base + cc]] = sys.rows[i].htt[[rr, cc]];
            }
            for cc in 0..k_beta {
                let v = sys.rows[i].htbeta[[rr, cc]];
                h[[base + rr, n * d + cc]] = v;
                h[[n * d + cc, base + rr]] = v;
            }
        }
    }
    for rr in 0..k_beta {
        for cc in 0..k_beta {
            h[[n * d + rr, n * d + cc]] = sys.hbb[[rr, cc]];
        }
    }
    // Cross-row i≠j terms: H[g_i, g_j] += d_k·z'_ik·z'_jk (the self i=j part
    // is already on the diagonal via the assembled self term).
    for k in 0..source.r {
        let dk = source.d[k];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let gi = i * d + k;
                let gj = j * d + k;
                h[[gi, gj]] += dk * zprime[i][k] * zprime[j][k];
            }
        }
    }
    h
}

#[test]
pub(crate) fn ibp_cross_row_woodbury_logdet_matches_dense() {
    let (mut sys, source, zprime) = build_ibp_woodbury_fixture();
    sys.set_ibp_cross_row_source(source.clone());
    let options = ArrowSolveOptions::direct();
    let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
        .expect("IBP Woodbury cache should factor");
    assert!(
        cache.cross_row_woodbury.is_some(),
        "the cache must carry the cross-row Woodbury"
    );

    let h_full = dense_h_full(&sys, &source, &zprime);
    let l = cholesky_lower(&h_full).expect("H_full must be SPD for this fixture");
    let mut dense_logdet = 0.0_f64;
    for i in 0..l.nrows() {
        dense_logdet += 2.0 * l[[i, i]].ln();
    }

    let (tt, schur) = cache.arrow_log_det();
    let cache_logdet = tt + schur.expect("direct mode has a Schur factor");
    assert!(
        (cache_logdet - dense_logdet).abs() < 1e-9,
        "cache log det H_full {cache_logdet} vs dense {dense_logdet}"
    );

    // The Woodbury correction is exactly log det H_full − log det H₀', where
    // the factored base `H₀' = H_full − U D Uᵀ` has the WHOLE rank-`R` update
    // removed — both the `i=j` self diagonal `d_k·z'_ik²` AND the `i≠j`
    // cross-row off-diagonals `d_k·z'_ik·z'_jk`. (The per-row latent blocks the
    // cache factors never carry cross-row coupling, so its base is exactly this
    // `H₀'`; subtracting only the self diagonal would leave the cross terms in
    // and compare the lemma correction against a different base.)
    let mut h0prime = h_full.clone();
    for k in 0..source.r {
        for i in 0..sys.rows.len() {
            let gi = i * sys.d + k;
            for j in 0..sys.rows.len() {
                let gj = j * sys.d + k;
                h0prime[[gi, gj]] -= source.d[k] * zprime[i][k] * zprime[j][k];
            }
        }
    }
    let l0 = cholesky_lower(&h0prime).expect("H₀' SPD");
    let mut logdet_h0prime = 0.0_f64;
    for i in 0..l0.nrows() {
        logdet_h0prime += 2.0 * l0[[i, i]].ln();
    }
    let correction = cache.cross_row_woodbury_log_det();
    assert!(
        (correction - (dense_logdet - logdet_h0prime)).abs() < 1e-9,
        "Woodbury log det correction {correction} vs (logdet H_full − logdet H₀') {}",
        dense_logdet - logdet_h0prime
    );
}

#[test]
pub(crate) fn ibp_cross_row_woodbury_full_inverse_and_newton_match_dense() {
    let (mut sys, source, zprime) = build_ibp_woodbury_fixture();
    sys.set_ibp_cross_row_source(source.clone());
    let options = ArrowSolveOptions::direct();
    let (delta_t, delta_beta, cache) =
        solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("IBP Woodbury cache should factor");

    let n = sys.rows.len();
    let d = sys.d;
    let k_beta = sys.k;
    let dim = n * d + k_beta;
    let h_full = dense_h_full(&sys, &source, &zprime);
    let l = cholesky_lower(&h_full).expect("H_full SPD");

    // (a) Newton step Δ = −H_full⁻¹ g.
    let mut g = Array1::<f64>::zeros(dim);
    for i in 0..n {
        for j in 0..d {
            g[i * d + j] = sys.rows[i].gt[j];
        }
    }
    for c in 0..k_beta {
        g[n * d + c] = sys.gb[c];
    }
    let dense_step = cholesky_solve_vector(&l, &g); // H_full⁻¹ g
    for idx in 0..n * d {
        assert!(
            (delta_t[idx] + dense_step[idx]).abs() < 1e-9,
            "Δt[{idx}] {} vs −H_full⁻¹g {}",
            delta_t[idx],
            -dense_step[idx]
        );
    }
    for c in 0..k_beta {
        assert!(
            (delta_beta[c] + dense_step[n * d + c]).abs() < 1e-9,
            "Δβ[{c}] {} vs −H_full⁻¹g {}",
            delta_beta[c],
            -dense_step[n * d + c]
        );
    }

    // (b) full_inverse_apply on an arbitrary RHS = dense H_full⁻¹ w.
    let mut w_full = Array1::<f64>::zeros(dim);
    for (idx, v) in w_full.iter_mut().enumerate() {
        *v = 0.25 + 0.13 * (idx as f64) * (if idx % 2 == 0 { 1.0 } else { -1.0 });
    }
    let dense_u = cholesky_solve_vector(&l, &w_full);
    let (u_t, u_beta) = cache
        .full_inverse_apply(
            w_full.slice(ndarray::s![..n * d]),
            w_full.slice(ndarray::s![n * d..]),
        )
        .expect("full_inverse_apply on the Woodbury cache");
    for idx in 0..n * d {
        assert!(
            (u_t[idx] - dense_u[idx]).abs() < 1e-9,
            "H_full⁻¹w t[{idx}] {} vs dense {}",
            u_t[idx],
            dense_u[idx]
        );
    }
    for c in 0..k_beta {
        assert!(
            (u_beta[c] - dense_u[n * d + c]).abs() < 1e-9,
            "H_full⁻¹w beta[{c}] {} vs dense {}",
            u_beta[c],
            dense_u[n * d + c]
        );
    }

    // (c) latent_block_inverse_diagonal = diag((H_full⁻¹)_tt).
    let mut h_full_inv = Array2::<f64>::zeros((dim, dim));
    let mut e = Array1::<f64>::zeros(dim);
    for col in 0..dim {
        e.fill(0.0);
        e[col] = 1.0;
        let sol = cholesky_solve_vector(&l, &e);
        for rrow in 0..dim {
            h_full_inv[[rrow, col]] = sol[rrow];
        }
    }
    let diag = cache
        .latent_block_inverse_diagonal()
        .expect("latent_block_inverse_diagonal on the Woodbury cache");
    for idx in 0..n * d {
        assert!(
            (diag[idx] - h_full_inv[[idx, idx]]).abs() < 1e-9,
            "diag (H_full⁻¹)_tt[{idx}] {} vs dense {}",
            diag[idx],
            h_full_inv[[idx, idx]]
        );
    }
}

/// Value↔gradient consistency: the log-determinant the evidence reports and
/// the Hessian the Newton/adjoint solve inverts must be the SAME `H_full`.
/// A finite-difference of `½ log det H(ε)` along the gradient direction
/// `g = ∂(½ wᵀ H_full w)/∂...` is overkill here; instead we verify the
/// cross-row correction's own internal coherence: removing the source must
/// recover the bare-`H₀'` log-determinant (no double-count), and the
/// rank-`R` capacitance LU determinant matches the dense ratio. (Covered by
/// `ibp_cross_row_woodbury_logdet_matches_dense`.) Here we additionally
/// check that a system WITHOUT the source yields no Woodbury carrier and an
/// unchanged (bare) log-determinant, so the path is a strict no-op off-IBP.
#[test]
pub(crate) fn ibp_cross_row_woodbury_absent_is_strict_noop() {
    let (sys, _source, zprime) = build_ibp_woodbury_fixture();
    // No set_ibp_cross_row_source call: the source is absent.
    let options = ArrowSolveOptions::direct();
    let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
        .expect("bare cache should factor");
    assert!(
        cache.cross_row_woodbury.is_none(),
        "no source ⇒ no Woodbury carrier"
    );
    assert_eq!(
        cache.cross_row_woodbury_log_det(),
        0.0,
        "absent Woodbury contributes exactly zero to the log-determinant"
    );
    // The bare cache's log det is that of the assembled self-term `H₀` (the
    // fixture's rows already carry the self term), with no cross-row terms.
    let dim = sys.rows.len() * sys.d + sys.k;
    let mut h0 = Array2::<f64>::zeros((dim, dim));
    let n = sys.rows.len();
    let d = sys.d;
    for i in 0..n {
        let base = i * d;
        for rr in 0..d {
            for cc in 0..d {
                h0[[base + rr, base + cc]] = sys.rows[i].htt[[rr, cc]];
            }
            for cc in 0..sys.k {
                let v = sys.rows[i].htbeta[[rr, cc]];
                h0[[base + rr, n * d + cc]] = v;
                h0[[n * d + cc, base + rr]] = v;
            }
        }
    }
    for rr in 0..sys.k {
        for cc in 0..sys.k {
            h0[[n * d + rr, n * d + cc]] = sys.hbb[[rr, cc]];
        }
    }
    let l = cholesky_lower(&h0).expect("H₀ SPD");
    let mut dense_logdet = 0.0_f64;
    for i in 0..l.nrows() {
        dense_logdet += 2.0 * l[[i, i]].ln();
    }
    let (tt, schur) = cache.arrow_log_det();
    let cache_logdet = tt + schur.expect("direct Schur");
    assert!(
        (cache_logdet - dense_logdet).abs() < 1e-9,
        "bare cache log det {cache_logdet} vs dense H₀ {dense_logdet}"
    );
    // `zprime` is part of the shared fixture; touch it so the helper's third
    // return stays meaningful for readers and is not dead in this arm.
    assert_eq!(zprime.len(), n);
}

/// The streaming log-det path must REFUSE an IBP-active system rather than
/// silently drop the cross-row correction (a value↔gradient desync).
#[test]
pub(crate) fn ibp_cross_row_streaming_logdet_refuses() {
    let (mut sys, source, _zprime) = build_ibp_woodbury_fixture();
    sys.set_ibp_cross_row_source(source);
    let mut streaming = StreamingArrowSchur::from_system(&sys, 2);
    let options = ArrowSolveOptions::direct();
    let err = streaming.reduced_schur_and_log_det_tt(0.0, 0.0, &options);
    assert!(
        err.is_err(),
        "streaming arrow log-det must refuse an IBP-active system"
    );
}

/// Build a dense-`htbeta` arrow system at an SAE-LLM-flavoured shape
/// (`n` row blocks × `d` latent coords × wide border `k`), with
/// deterministic well-conditioned per-row blocks and cross-blocks. This is
/// the shape the reduced-Schur matvec (#1017) walks O(cg_iters) times.
pub(crate) fn dense_arrow_system(n: usize, d: usize, k: usize) -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(n, d, k);
    // Deterministic diagonally-dominant per-row H_tt and modest H_tβ.
    for i in 0..n {
        let mut htt = Array2::<f64>::zeros((d, d));
        for r in 0..d {
            for c in 0..d {
                let s = ((i + 1) * (r + 2) * (c + 3)) as f64;
                htt[[r, c]] = if r == c {
                    4.0 + (s % 7.0)
                } else {
                    0.1 * ((s % 5.0) - 2.0)
                };
            }
        }
        // Symmetrize and ensure SPD by diagonal dominance.
        let mut sym = &htt + &htt.t();
        for r in 0..d {
            sym[[r, r]] = sym[[r, r]].abs() + (d as f64) + 2.0;
        }
        sys.rows[i].htt = sym;
        let mut htb = Array2::<f64>::zeros((d, k));
        for r in 0..d {
            for c in 0..k {
                let s = ((i + 1) * (r + 1) + 3 * (c + 1)) as f64;
                htb[[r, c]] = 0.05 * ((s % 11.0) - 5.0);
            }
        }
        sys.rows[i].htbeta = htb;
        sys.rows[i].gt = Array1::<f64>::zeros(d);
    }
    // SPD H_ββ: diagonally dominant.
    let mut hbb = Array2::<f64>::zeros((k, k));
    for r in 0..k {
        for c in 0..k {
            let s = ((r + 1) * (c + 1)) as f64;
            hbb[[r, c]] = if r == c {
                (k as f64) + 6.0 + (s % 3.0)
            } else {
                0.02 * ((s % 7.0) - 3.0)
            };
        }
    }
    sys.hbb = hbb;
    sys.gb = Array1::<f64>::zeros(k);
    sys
}

/// Sequential reference for the reduced-Schur matvec: the exact per-row fold
/// the `schur_matvec` sequential branch performs (used to compare the
/// parallel path against). Mirrors the production routine's H_ββ + ridge
/// prologue, then the per-row point-elimination subtraction in row order.
pub(crate) fn schur_matvec_sequential_ref<B: BatchedBlockSolver>(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    ridge_beta: f64,
    x: &Array1<f64>,
    backend: &B,
) -> Array1<f64> {
    let k = sys.k;
    let mut out = Array1::<f64>::zeros(k);
    {
        let xs = x.as_slice().unwrap();
        let os = out.as_slice_mut().unwrap();
        sys.penalty_matvec_add(xs, os);
        for a in 0..k {
            os[a] += ridge_beta * xs[a];
        }
    }
    let mut local = Array1::<f64>::zeros(sys.d);
    let mut neg = Array1::<f64>::zeros(k);
    for i in 0..sys.rows.len() {
        neg.fill(0.0);
        schur_matvec_row_into(sys, htt_factors, x, backend, i, &mut local, &mut neg);
        for a in 0..k {
            out[a] -= neg[a];
        }
    }
    out
}

/// The parallel reduced-Schur matvec (rows ≥ `SCHUR_MATVEC_PARALLEL_ROW_MIN`)
/// must be (a) DETERMINISTIC run-to-run — bit-identical across repeated
/// invocations regardless of thread scheduling, the #1017 verification gate
/// that the criterion ranking across candidates cannot move; and (b)
/// numerically equal to the sequential per-row fold up to the ULP-level
/// reordering of an otherwise-identical sum (the chunk-partial reduction
/// reassociates the same row contributions, so it agrees with the per-row
/// fold to a tight relative tolerance, not bit-for-bit).
#[test]
pub(crate) fn parallel_schur_matvec_deterministic_and_matches_sequential() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64; // trips the parallel path
    let d = 6usize;
    let k = 96usize;
    let sys = dense_arrow_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;
    let x = Array1::from_iter((0..k).map(|a| 0.3 * (a as f64).sin() - 0.1));

    // (a) Determinism: two independent invocations of the live (parallel)
    // path must be bit-identical.
    let mut out_a = Array1::<f64>::zeros(k);
    let mut out_b = Array1::<f64>::zeros(k);
    schur_matvec(
        &sys,
        &htt_factors,
        ridge_beta,
        &x,
        &mut out_a,
        &backend,
        None,
    );
    schur_matvec(
        &sys,
        &htt_factors,
        ridge_beta,
        &x,
        &mut out_b,
        &backend,
        None,
    );
    for a in 0..k {
        assert_eq!(
            out_a[a].to_bits(),
            out_b[a].to_bits(),
            "parallel Schur matvec must be deterministic run-to-run at index {a}"
        );
    }

    // (b) Equivalence with the sequential per-row fold within ULP-scale
    // reassociation error.
    let out_seq = schur_matvec_sequential_ref(&sys, &htt_factors, ridge_beta, &x, &backend);
    let scale = out_seq
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    for a in 0..k {
        let rel = (out_a[a] - out_seq[a]).abs() / scale;
        assert!(
            rel < 1e-12,
            "parallel vs sequential Schur matvec must agree to reassociation error \
                 at index {a}: {} vs {} (rel {rel:e})",
            out_a[a],
            out_seq[a]
        );
    }
}

/// Sequential reference for the cross-row matvec: the row-order fold of the
/// same per-row contributions `arrow_cross_row_matvec` accumulates, followed by
/// the post-loop `H_ββ + ridge` prologue and cross-row penalty Hessian. Used to
/// pin the parallelized n-row loop against an independent serial computation.
pub(crate) fn cross_row_matvec_sequential_ref(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
    x_t: ArrayView1<'_, f64>,
    x_beta: ArrayView1<'_, f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = sys.rows.len();
    let k = sys.k;
    let total_dt = sys.row_offsets[n];
    let mut y_t = Array1::<f64>::zeros(total_dt);
    let mut y_beta = Array1::<f64>::zeros(k);
    for i in 0..n {
        let di = sys.row_dims[i];
        let base = sys.row_offsets[i];
        let row = &sys.rows[i];
        for a in 0..di {
            let mut acc = ridge_t * x_t[base + a];
            for b in 0..di {
                acc += row.htt[[a, b]] * x_t[base + b];
            }
            y_t[base + a] = acc;
        }
        let mut slab = Array1::<f64>::zeros(di);
        sys_htbeta_apply_row(sys, i, row, x_beta, &mut slab);
        for c in 0..di {
            y_t[base + c] += slab[c];
        }
        let x_ti = x_t.slice(ndarray::s![base..base + di]).to_owned();
        sys_htbeta_accumulate_transpose(sys, i, row, x_ti.view(), &mut y_beta);
    }
    {
        let x_beta_slice = x_beta.as_slice().expect("x_beta contiguous");
        let y_beta_slice = y_beta.as_slice_mut().expect("y_beta contiguous");
        sys.penalty_matvec_add(x_beta_slice, y_beta_slice);
    }
    for a in 0..k {
        y_beta[a] += ridge_beta * x_beta[a];
    }
    sys.apply_cross_row_penalty_hessian(x_t, &mut y_t);
    (y_t, y_beta)
}

/// The parallel cross-row matvec (`arrow_cross_row_matvec`, the per-CG-iteration
/// operator of the cross-row coupled Newton solve) must, like its `schur_matvec`
/// twin, be (a) DETERMINISTIC run-to-run — bit-identical across repeated
/// invocations regardless of thread scheduling (the #1017 gate: the criterion
/// ranking across candidates cannot move); and (b) equal to the sequential
/// row-order fold — bit-identical on the disjoint `y_t` writes and within
/// ULP-scale reassociation on the cross-row `y_beta` sum.
#[test]
pub(crate) fn parallel_cross_row_matvec_deterministic_and_matches_sequential() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 96; // trips the parallel path
    let d = 5usize;
    let k = 80usize;
    let sys = dense_arrow_system(n, d, k);
    let total_dt = sys.row_offsets[n];
    let ridge_t = 1e-5;
    let ridge_beta = 1e-6;
    let x_t = Array1::from_iter((0..total_dt).map(|i| 0.2 * (i as f64).cos() + 0.05));
    let x_beta = Array1::from_iter((0..k).map(|a| 0.3 * (a as f64).sin() - 0.1));

    // (a) Determinism: two independent invocations of the live (parallel) path
    // must be bit-identical in both output blocks.
    let (yt_a, yb_a) = arrow_cross_row_matvec(&sys, ridge_t, ridge_beta, x_t.view(), x_beta.view());
    let (yt_b, yb_b) = arrow_cross_row_matvec(&sys, ridge_t, ridge_beta, x_t.view(), x_beta.view());
    for i in 0..total_dt {
        assert_eq!(
            yt_a[i].to_bits(),
            yt_b[i].to_bits(),
            "parallel cross-row matvec y_t must be deterministic at {i}"
        );
    }
    for a in 0..k {
        assert_eq!(
            yb_a[a].to_bits(),
            yb_b[a].to_bits(),
            "parallel cross-row matvec y_beta must be deterministic at {a}"
        );
    }

    // (b) Equivalence with the sequential row-order fold.
    let (yt_seq, yb_seq) =
        cross_row_matvec_sequential_ref(&sys, ridge_t, ridge_beta, x_t.view(), x_beta.view());
    // y_t writes are disjoint per row → bit-identical to the serial fold.
    for i in 0..total_dt {
        assert_eq!(
            yt_a[i].to_bits(),
            yt_seq[i].to_bits(),
            "parallel cross-row matvec y_t must match the sequential fold bit-for-bit at {i}"
        );
    }
    // y_beta is a cross-row accumulation → equal within reassociation error.
    let scale = yb_seq.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
    for a in 0..k {
        let rel = (yb_a[a] - yb_seq[a]).abs() / scale;
        assert!(
            rel < 1e-12,
            "parallel vs sequential cross-row matvec y_beta must agree to reassociation \
                 error at {a}: {} vs {} (rel {rel:e})",
            yb_a[a],
            yb_seq[a]
        );
    }
}

/// The cross-row preconditioner solve `ArrowBlockDiagInverse::apply` (run once
/// per cross-row CG iteration) parallelizes both its n-row passes (#1017). It
/// must be (a) DETERMINISTIC run-to-run and (b) the exact inverse of the
/// block-diagonal arrow operator `K0 + ridge`. With no cross-row penalties
/// `P_cross = 0`, so `arrow_cross_row_matvec` IS `K0 + ridge`; the round trip
/// `(K0+ridge)·apply(r)` must recover `r`.
#[test]
pub(crate) fn parallel_block_diag_inverse_apply_deterministic_and_solves() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64; // trips the parallel path
    let d = 4usize;
    let k = 72usize;
    let sys = dense_arrow_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_t = 1e-4;
    let ridge_beta = 1e-5;
    let precond = ArrowBlockDiagInverse::build(&sys, ridge_t, ridge_beta, false, &backend)
        .expect("block-diagonal inverse must build");
    let total_dt = sys.row_offsets[n];
    let r_t = Array1::from_iter((0..total_dt).map(|i| 0.15 * (i as f64).sin() + 0.02));
    let r_beta = Array1::from_iter((0..k).map(|a| 0.25 * (a as f64).cos() - 0.05));

    // (a) Determinism run-to-run on the parallel path.
    let (xt_a, xb_a) = precond.apply(r_t.view(), r_beta.view());
    let (xt_b, xb_b) = precond.apply(r_t.view(), r_beta.view());
    for i in 0..total_dt {
        assert_eq!(
            xt_a[i].to_bits(),
            xt_b[i].to_bits(),
            "preconditioner x_t must be deterministic at {i}"
        );
    }
    for a in 0..k {
        assert_eq!(
            xb_a[a].to_bits(),
            xb_b[a].to_bits(),
            "preconditioner x_beta must be deterministic at {a}"
        );
    }

    // (b) Exact inverse: the round trip recovers the RHS.
    let (yt, yb) = arrow_cross_row_matvec(&sys, ridge_t, ridge_beta, xt_a.view(), xb_a.view());
    let scale_t = r_t.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
    for i in 0..total_dt {
        let rel = (yt[i] - r_t[i]).abs() / scale_t;
        assert!(
            rel < 1e-9,
            "preconditioner round-trip y_t at {i}: rel {rel:e}"
        );
    }
    let scale_b = r_beta.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
    for a in 0..k {
        let rel = (yb[a] - r_beta[a]).abs() / scale_b;
        assert!(
            rel < 1e-9,
            "preconditioner round-trip y_beta at {a}: rel {rel:e}"
        );
    }
}

/// `arrow_operator_apply` (the block-diagonal `K0` operator used by the
/// iterative-refinement residual / backward-error certificate) parallelizes its
/// n-row pass via the shared `cross_row_matvec_row_into` body (#1017). It must
/// be deterministic run-to-run and equal to the sequential fold: with no
/// cross-row penalties it equals `arrow_cross_row_matvec`, so the same
/// `cross_row_matvec_sequential_ref` is the reference (bit-identical disjoint
/// `y_t`, ULP-scale `y_beta` reassociation).
#[test]
pub(crate) fn parallel_arrow_operator_apply_deterministic_and_matches_sequential() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 48; // trips the parallel path
    let d = 6usize;
    let k = 64usize;
    let sys = dense_arrow_system(n, d, k);
    let total_dt = sys.row_offsets[n];
    let ridge_t = 2e-5;
    let ridge_beta = 3e-6;
    let x_t = Array1::from_iter((0..total_dt).map(|i| 0.17 * (i as f64).sin() - 0.03));
    let x_beta = Array1::from_iter((0..k).map(|a| 0.21 * (a as f64).cos() + 0.04));

    let (yt_a, yb_a) = arrow_operator_apply(&sys, ridge_t, ridge_beta, x_t.view(), x_beta.view());
    let (yt_b, yb_b) = arrow_operator_apply(&sys, ridge_t, ridge_beta, x_t.view(), x_beta.view());
    for i in 0..total_dt {
        assert_eq!(
            yt_a[i].to_bits(),
            yt_b[i].to_bits(),
            "arrow_operator_apply y_t must be deterministic at {i}"
        );
    }
    for a in 0..k {
        assert_eq!(
            yb_a[a].to_bits(),
            yb_b[a].to_bits(),
            "arrow_operator_apply y_beta must be deterministic at {a}"
        );
    }

    let (yt_seq, yb_seq) =
        cross_row_matvec_sequential_ref(&sys, ridge_t, ridge_beta, x_t.view(), x_beta.view());
    for i in 0..total_dt {
        assert_eq!(
            yt_a[i].to_bits(),
            yt_seq[i].to_bits(),
            "arrow_operator_apply y_t must match the sequential fold bit-for-bit at {i}"
        );
    }
    let scale = yb_seq.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
    for a in 0..k {
        let rel = (yb_a[a] - yb_seq[a]).abs() / scale;
        assert!(
            rel < 1e-12,
            "arrow_operator_apply y_beta vs sequential at {a}: rel {rel:e}"
        );
    }
}

/// The dense `H_ββ` penalty-prologue GEMV parallelized over output rows at
/// the wide SAE border (`k ≥ SCHUR_PROLOGUE_PARALLEL_K_MIN`, #1017) must be
/// **bit-identical** to the serial prologue — unlike the per-row reduction,
/// the GEMV carries no reassociation: each `y[a] = Σ_b hbb[a,b]·x[b] + ridge·x[a]`
/// is computed in its entirety by one thread in the same `b` order whether
/// one core or many run, so distributing the `a`-rows across threads cannot
/// move a single bit. This pins the determinism/parity gate exactly at the
/// border width where the prologue stops being serial.
#[test]
pub(crate) fn parallel_penalty_prologue_bit_identical_to_serial() {
    let k = 576usize; // ≥ SCHUR_PROLOGUE_PARALLEL_K_MIN: trips the parallel GEMV
    assert!(
        k >= SCHUR_PROLOGUE_PARALLEL_K_MIN,
        "test border must exceed the prologue parallel threshold"
    );
    let d = 4usize;
    // A handful of rows: small enough that the per-row loop stays sequential
    // (rows < SCHUR_MATVEC_PARALLEL_ROW_MIN), isolating the prologue as the
    // only parallelized stage so the bit-parity claim is about it alone.
    let n = 8usize;
    assert!(n < SCHUR_MATVEC_PARALLEL_ROW_MIN);
    let sys = dense_arrow_system(n, d, k);
    let ridge = 7.5e-3;
    let x = Array1::from_iter((0..k).map(|a| 0.4 * (a as f64 * 0.31).cos() - 0.17));
    let xs = x.as_slice().unwrap();

    // Serial reference: penalty_matvec_add + ridge axpy into a zeroed buffer.
    let mut serial = vec![0.0_f64; k];
    sys.penalty_matvec_add(xs, &mut serial);
    for a in 0..k {
        serial[a] += ridge * xs[a];
    }

    // Parallel prologue (parallel=true engages the rayon dense GEMV at this k).
    let mut par = vec![0.0_f64; k];
    sys.penalty_ridge_prologue_into(xs, ridge, &mut par, true);
    // And the serial branch of the same fn (parallel=false) for completeness.
    let mut ser_branch = vec![0.0_f64; k];
    sys.penalty_ridge_prologue_into(xs, ridge, &mut ser_branch, false);

    for a in 0..k {
        assert_eq!(
            par[a].to_bits(),
            serial[a].to_bits(),
            "parallel penalty prologue must be bit-identical to serial at index {a}"
        );
        assert_eq!(
            ser_branch[a].to_bits(),
            serial[a].to_bits(),
            "serial prologue branch must match the reference at index {a}"
        );
    }
}

/// Wall-clock benchmark of the reduced-Schur matvec at an SAE-LLM-flavoured
/// shape (#1017): sequential per-row fold vs the rayon-parallel chunked path.
/// Runs as an ordinary test (the ban gate forbids `#[ignore]`), so the shape
/// and call count are sized to stay fast in a debug CI build while still
/// tripping the parallel path. Run with `--release --nocapture` on a quiet
/// multicore box to read the per-call wall-clock and the parallel speedup at
/// the inner-CG matvec cost the production InexactPCG loop pays O(cg_iters)
/// times:
///
/// ```text
/// cargo test -p gam --lib --release \
///   solver::arrow_schur::tests::bench_reduced_schur_matvec_parallel_speedup \
///   -- --nocapture
/// ```
#[test]
pub(crate) fn bench_reduced_schur_matvec_parallel_speedup() {
    // SAE-arm-flavoured shape from the issue: many row blocks, wide border
    // k, modest frame depth d. Sized so the debug build stays quick.
    let n = 1500usize;
    let d = 6usize;
    let k = 1024usize;
    let sys = dense_arrow_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;
    let x = Array1::from_iter((0..k).map(|a| 0.3 * ((a as f64) * 0.017).sin() - 0.1));

    // A representative inner-CG budget: the matvec is paid once per CG iter.
    let calls = 30usize;
    let mut sink = 0.0_f64;

    // Warm up (factor caches, allocator, rayon pool) before timing.
    let warm = schur_matvec_sequential_ref(&sys, &htt_factors, ridge_beta, &x, &backend);
    sink += warm[0];

    let t_seq = std::time::Instant::now();
    for _ in 0..calls {
        let out = schur_matvec_sequential_ref(&sys, &htt_factors, ridge_beta, &x, &backend);
        sink += out[0];
    }
    let seq_elapsed = t_seq.elapsed();

    let mut out_par = Array1::<f64>::zeros(k);
    schur_matvec(
        &sys,
        &htt_factors,
        ridge_beta,
        &x,
        &mut out_par,
        &backend,
        None,
    ); // warm
    sink += out_par[0];
    let t_par = std::time::Instant::now();
    for _ in 0..calls {
        schur_matvec(
            &sys,
            &htt_factors,
            ridge_beta,
            &x,
            &mut out_par,
            &backend,
            None,
        );
        sink += out_par[0];
    }
    let par_elapsed = t_par.elapsed();

    let seq_per = seq_elapsed.as_secs_f64() / calls as f64;
    let par_per = par_elapsed.as_secs_f64() / calls as f64;
    let speedup = seq_per / par_per;
    println!(
        "[#1017 reduced-Schur matvec, n={n} d={d} k={k}, {calls} calls, \
             {} rayon threads]\n  sequential: {:.3} ms/call\n  parallel:   {:.3} ms/call\n  \
             speedup:    {:.2}x  (sink {:.3e})",
        rayon::current_num_threads(),
        seq_per * 1e3,
        par_per * 1e3,
        speedup,
        sink,
    );
    // Loose floor so a single-core or heavily-loaded box does not flap the
    // benchmark; the real signal is the printed numbers.
    assert!(par_per > 0.0 && seq_per > 0.0, "timings must be positive");
}

/// Build an SAE-structured arrow system exercising the residency path: per
/// row a `q×q` SPD `H_tt`, a `q×p` local Jacobian `L_i`, and `m_i` active
/// atoms over `n_atoms` decoder blocks of width `p` (border `k = n_atoms·p`).
/// Installs BOTH the matrix-free Kronecker cross-block operator (the generic
/// matvec path: `H_tβ = L_i P_i`) AND the matching `DeviceSaePcgData` (the
/// residency path), so the two routes see the identical operator.
pub(crate) fn sae_structured_system(
    n: usize,
    q: usize,
    p: usize,
    n_atoms: usize,
    m_active: usize,
) -> (ArrowSchurSystem, Vec<Vec<(usize, f64)>>, Vec<Vec<f64>>) {
    let k = n_atoms * p;
    let mut sys = ArrowSchurSystem::new(n, q, k);
    let mut a_phi: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    let mut local_jac: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        // SPD H_tt: diagonally dominant.
        let mut htt = Array2::<f64>::zeros((q, q));
        for r in 0..q {
            for c in 0..q {
                let s = ((i + 1) * (r + 2) * (c + 3)) as f64;
                htt[[r, c]] = 0.1 * ((s % 5.0) - 2.0);
            }
        }
        let mut sym = &htt + &htt.t();
        for r in 0..q {
            sym[[r, r]] = sym[[r, r]].abs() + (q as f64) + 3.0;
        }
        sys.rows[i].htt = sym;
        sys.rows[i].gt = Array1::<f64>::zeros(q);
        // L_i (q×p), row-major.
        let mut jac = vec![0.0_f64; q * p];
        for c in 0..q {
            for j in 0..p {
                let s = ((i + 1) + 2 * (c + 1) + 3 * (j + 1)) as f64;
                jac[c * p + j] = 0.1 * ((s % 7.0) - 3.0);
            }
        }
        local_jac.push(jac);
        // m_active atoms per row, deterministic spread over n_atoms.
        let mut support = Vec::with_capacity(m_active);
        for s in 0..m_active {
            let atom = ((i * 3 + s * 5) % n_atoms).min(n_atoms - 1);
            let phi = 0.5 + 0.25 * (((i + s) % 4) as f64);
            support.push((atom * p, phi));
        }
        a_phi.push(support);
    }
    // SPD H_ββ.
    let mut hbb = Array2::<f64>::zeros((k, k));
    for r in 0..k {
        hbb[[r, r]] = (k as f64) + 4.0;
    }
    sys.hbb = hbb;
    sys.gb = Array1::<f64>::zeros(k);
    // Install the matrix-free Kronecker operator (H_tβ = L_i · P_i): forward
    // gathers active atoms into a length-p vector then applies L_i; transpose
    // is the exact adjoint. Mirrors src/terms/sae_manifold.rs:6028.
    let a_phi_f = a_phi.clone();
    let jac_f = local_jac.clone();
    let a_phi_t = a_phi.clone();
    let jac_t = local_jac.clone();
    let p_f = p;
    sys.set_row_htbeta_operator(
        move |row, x, out| {
            let mut u_p = vec![0.0_f64; p_f];
            for &(base, phi) in &a_phi_f[row] {
                for j in 0..p_f {
                    u_p[j] += phi * x[base + j];
                }
            }
            let jac = &jac_f[row];
            let qi = jac.len() / p_f;
            for c in 0..qi {
                let mut acc = 0.0;
                for j in 0..p_f {
                    acc += jac[c * p_f + j] * u_p[j];
                }
                out[c] = acc;
            }
        },
        move |row, v, out| {
            let jac = &jac_t[row];
            let qi = jac.len() / p_f;
            let mut u_p = vec![0.0_f64; p_f];
            for c in 0..qi {
                let vc = v[c];
                for j in 0..p_f {
                    u_p[j] += jac[c * p_f + j] * vc;
                }
            }
            for &(base, phi) in &a_phi_t[row] {
                for j in 0..p_f {
                    out[base + j] += phi * u_p[j];
                }
            }
        },
    );
    sys.set_device_sae_pcg_data(DeviceSaePcgData {
        p,
        beta_dim: k,
        a_phi: a_phi.clone(),
        local_jac: local_jac.clone(),
        smooth_blocks: Vec::new(),
        sparse_g_blocks: Vec::new(),
    });
    (sys, a_phi, local_jac)
}

/// The CPU-resident SAE reduced-Schur matvec (#1017) must compute the SAME
/// `S·x` as the generic per-row `apply → solve → transpose` path, up to f64
/// reassociation. This is the residency correctness gate: a resident matvec
/// that changed the reduced operator would change the Newton step and the
/// criterion ranking — a correctness regression, not a speedup.
#[test]
pub(crate) fn resident_sae_matvec_matches_generic() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 96; // trips the parallel path
    let q = 4usize;
    let p = 6usize;
    let n_atoms = 32usize;
    let m_active = 5usize;
    let (sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
    let k = sys.k;
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, q, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;
    let x = Array1::from_iter((0..k).map(|a| 0.2 * ((a as f64) * 0.013).cos() - 0.05));

    // Generic path (no resident operator).
    let mut out_generic = Array1::<f64>::zeros(k);
    schur_matvec(
        &sys,
        &htt_factors,
        ridge_beta,
        &x,
        &mut out_generic,
        &backend,
        None,
    );

    // Resident path: stage G_i once, then matvec.
    let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
        .expect("SAE structure must yield a resident operator");
    let mut out_resident = Array1::<f64>::zeros(k);
    schur_matvec(
        &sys,
        &htt_factors,
        ridge_beta,
        &x,
        &mut out_resident,
        &backend,
        Some(&resident),
    );

    let scale = out_generic
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    for a in 0..k {
        let rel = (out_resident[a] - out_generic[a]).abs() / scale;
        assert!(
            rel < 1e-10,
            "resident vs generic SAE Schur matvec must agree at index {a}: \
                 {} vs {} (rel {rel:e})",
            out_resident[a],
            out_generic[a]
        );
    }

    // Determinism: rebuilding + re-applying is bit-identical run-to-run.
    let resident2 = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend).unwrap();
    let mut out_resident2 = Array1::<f64>::zeros(k);
    schur_matvec(
        &sys,
        &htt_factors,
        ridge_beta,
        &x,
        &mut out_resident2,
        &backend,
        Some(&resident2),
    );
    for a in 0..k {
        assert_eq!(
            out_resident[a].to_bits(),
            out_resident2[a].to_bits(),
            "resident SAE matvec must be deterministic run-to-run at index {a}"
        );
    }
}

/// The #1017 SAE-resident scalar Jacobi (built from the staged `(L_i, Y_i)`
/// factors in one support-sparse pass) must produce the SAME reduced-Schur
/// diagonal — hence the SAME `BlockFactor::Scalar` inverses — as the generic
/// per-column probe-and-solve `build_scalar_jacobi`. A diverging
/// preconditioner would change the PCG iterate and the criterion ranking.
#[test]
pub(crate) fn resident_scalar_jacobi_matches_generic() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64;
    let q = 4usize;
    let p = 5usize;
    let n_atoms = 20usize;
    let m_active = 4usize;
    let (sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, q, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;

    let generic =
        JacobiPreconditioner::build_scalar_jacobi(&sys, &htt_factors, ridge_beta, &backend)
            .expect("generic scalar Jacobi must build");
    let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
        .expect("SAE structure must yield a resident operator");
    let resident_jac =
        JacobiPreconditioner::build_scalar_jacobi_resident(&sys, ridge_beta, &resident)
            .expect("resident scalar Jacobi must build");

    // Probe both preconditioners with the same residual and compare the
    // applied (diagonal-scaled) output: identical diagonals ⇒ identical apply.
    let k = sys.k;
    let r = Array1::from_iter((0..k).map(|a| 0.3 * ((a as f64) * 0.021).sin() + 0.07));
    let out_generic = generic.apply(&r);
    let out_resident = resident_jac.apply(&r);
    let scale = out_generic
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    for a in 0..k {
        let rel = (out_resident[a] - out_generic[a]).abs() / scale;
        assert!(
            rel < 1e-9,
            "resident vs generic SAE scalar Jacobi must agree at index {a}: \
                 {} vs {} (rel {rel:e})",
            out_resident[a],
            out_generic[a]
        );
    }
}

/// The factored residency (storing `(L_i, Y_i)` and applying `G_i v =
/// L_iᵀ(Y_i v)`) must reproduce the dense `p×p` block `G_i = L_iᵀ Y_i`
/// exactly — this is the #1017 memory/compute win (`O(n·di·p)` vs `O(n·p²)`)
/// and must not perturb the operator. Asserts, per row, that the factored
/// `row_into` applied to a unit-support probe equals the explicit dense
/// `G_i · (P_i x)` to rel < 1e-10.
#[test]
pub(crate) fn factored_residency_matches_dense_g_block() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 40;
    let q = 3usize;
    let p = 7usize;
    let n_atoms = 24usize;
    let m_active = 4usize;
    let (sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, q, false)
        .expect("SPD per-row blocks must factor");
    let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
        .expect("SAE structure must yield a resident operator");

    for row in 0..n {
        let rf = &resident.rows[row];
        if rf.di == 0 {
            continue;
        }
        let di = rf.di;
        // Reconstruct the dense block G_i = L_iᵀ Y_i (p×p) from the stored
        // factors and check the factored GEMV chain against a direct G_i·g.
        let l = ArrayView2::from_shape((di, p), &rf.l).unwrap();
        let y = ArrayView2::from_shape((di, p), &rf.y).unwrap();
        let g_dense = l.t().dot(&y); // p×p

        // A non-trivial gather vector g (length p).
        let g_vec: Vec<f64> = (0..p)
            .map(|j| 0.4 * ((row + j) as f64 * 0.11).sin() - 0.07)
            .collect();
        // Dense reference: prod_ref = G_i · g.
        let mut prod_ref = vec![0.0_f64; p];
        for r in 0..p {
            let mut s = 0.0;
            for c in 0..p {
                s += g_dense[(r, c)] * g_vec[c];
            }
            prod_ref[r] = s;
        }
        // Factored chain: w = Y_i·g, prod = L_iᵀ·w.
        let mut w = vec![0.0_f64; di];
        for r in 0..di {
            let yrow = &rf.y[r * p..r * p + p];
            w[r] = (0..p).map(|c| yrow[c] * g_vec[c]).sum();
        }
        let mut prod = vec![0.0_f64; p];
        for r in 0..di {
            let lrow = &rf.l[r * p..r * p + p];
            for j in 0..p {
                prod[j] += lrow[j] * w[r];
            }
        }
        let scale = prod_ref
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.abs()))
            .max(1.0);
        for j in 0..p {
            let rel = (prod[j] - prod_ref[j]).abs() / scale;
            assert!(
                rel < 1e-10,
                "factored G_i apply must match dense G_i at row {row} idx {j}: \
                     {} vs {} (rel {rel:e})",
                prod[j],
                prod_ref[j]
            );
        }
    }
    // Storage check: the factored form keeps di·p (not p²) per row.
    let factored_entries: usize = resident.rows.iter().map(|r| r.l.len() + r.y.len()).sum();
    let dense_entries: usize = resident.rows.iter().filter(|r| r.di > 0).count() * p * p;
    assert!(
        factored_entries < dense_entries,
        "factored residency must store fewer entries than the dense p×p form \
             ({factored_entries} vs {dense_entries})"
    );
}

/// Wall-clock benchmark: generic per-row matvec vs the CPU-resident SAE
/// matvec (#1017) at an SAE-flavoured shape, amortised over a representative
/// CG-iteration count (the residency build is paid once, then N matvecs).
/// Ordinary test (ban gate forbids `#[ignore]`); run `--release --nocapture`.
#[test]
pub(crate) fn bench_resident_sae_matvec_speedup() {
    // SAE shape: small per-row latent dim `q = di` (1–2 in production) and a
    // wider per-atom decoder block `p` — the regime where the factored
    // residency (`2·di·p` flops/row, `O(n·di·p)` memory) beats both the
    // generic per-iteration solve AND a dense `p×p` residency (`p²` /
    // `O(n·p²)`). Here di=2, p=64 ⇒ ~16× fewer matvec flops/row than dense.
    let n = 1500usize;
    let q = 2usize;
    let p = 64usize;
    let n_atoms = 32usize; // border k = n_atoms·p = 2048
    let m_active = 6usize;
    let (sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
    let k = sys.k;
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, q, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;
    let x = Array1::from_iter((0..k).map(|a| 0.3 * ((a as f64) * 0.017).sin() - 0.1));
    let cg_iters = 30usize;
    let mut sink = 0.0_f64;

    // Generic: matvec re-walks apply/solve/transpose every iteration.
    let mut out = Array1::<f64>::zeros(k);
    schur_matvec(&sys, &htt_factors, ridge_beta, &x, &mut out, &backend, None); // warm
    sink += out[0];
    let t_gen = std::time::Instant::now();
    for _ in 0..cg_iters {
        schur_matvec(&sys, &htt_factors, ridge_beta, &x, &mut out, &backend, None);
        sink += out[0];
    }
    let gen_elapsed = t_gen.elapsed();

    // Resident: stage once (timed into the total — honest amortisation),
    // then cg_iters cheap matvecs.
    let t_res = std::time::Instant::now();
    let resident =
        SaeResidentReducedSchur::build(&sys, &htt_factors, &backend).expect("resident operator");
    let mut outr = Array1::<f64>::zeros(k);
    for _ in 0..cg_iters {
        schur_matvec(
            &sys,
            &htt_factors,
            ridge_beta,
            &x,
            &mut outr,
            &backend,
            Some(&resident),
        );
        sink += outr[0];
    }
    let res_elapsed = t_res.elapsed();

    let gen_total = gen_elapsed.as_secs_f64();
    let res_total = res_elapsed.as_secs_f64();
    // Residency footprint: factored `(L_i, Y_i)` = `2·di·p` f64/row vs the
    // dense `p×p` block = `p²` f64/row.
    let factored_f64: usize = resident.rows.iter().map(|r| r.l.len() + r.y.len()).sum();
    let dense_f64: usize = resident.rows.iter().filter(|r| r.di > 0).count() * p * p;
    println!(
        "[#1017 SAE resident matvec, n={n} q={q} p={p} k={k} m={m_active}, \
             {cg_iters} CG matvecs incl. 1 residency build, {} rayon threads]\n  \
             generic:  {:.3} ms total ({:.3} ms/matvec)\n  resident: {:.3} ms total \
             (build + {cg_iters} matvecs)\n  speedup:  {:.2}x  (sink {:.3e})\n  \
             residency mem: factored {:.2} MiB vs dense p×p {:.2} MiB ({:.1}× smaller)",
        rayon::current_num_threads(),
        gen_total * 1e3,
        gen_total / cg_iters as f64 * 1e3,
        res_total * 1e3,
        gen_total / res_total,
        sink,
        factored_f64 as f64 * 8.0 / (1024.0 * 1024.0),
        dense_f64 as f64 * 8.0 / (1024.0 * 1024.0),
        dense_f64 as f64 / factored_f64.max(1) as f64,
    );
    assert!(
        gen_total > 0.0 && res_total > 0.0,
        "timings must be positive"
    );
}

/// #1017 streaming-assembly parallelism: `accumulate_chunk` (reduced-Schur +
/// reduced-RHS assembly) and `back_substitute` (per-row `Δt_i`) fan over rows
/// with rayon above `SCHUR_MATVEC_PARALLEL_ROW_MIN`. Both must be
/// (a) DETERMINISTIC run-to-run — bit-identical regardless of thread
/// scheduling, the #1017 verification gate that the criterion ranking cannot
/// move; and (b) numerically equal to the sequential per-row computation up to
/// ULP-level reassociation (the chunk-partial fold reassociates the SAME row
/// contributions). For `back_substitute` the per-row writes are DISJOINT, so it
/// must match the sequential scatter bit-for-bit.
#[test]
pub(crate) fn parallel_streaming_assembly_deterministic_and_matches_sequential() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64; // trips the parallel path
    let d = 4usize;
    let k = 24usize;
    let sys = dense_direct_system(n, d, k);
    let options = ArrowSolveOptions::direct();

    // (a) Determinism: two independent full solves at the parallel shape must
    // be bit-identical (Δt, Δβ, and the reduced Schur factor diagonal).
    let mut s_a = StreamingArrowSchur::from_system(&sys, n); // one big chunk → parallel accumulate
    let (dt_a, db_a, _) = s_a.solve(0.0, 0.0, &options).expect("parallel solve a");
    let mut s_b = StreamingArrowSchur::from_system(&sys, n);
    let (dt_b, db_b, _) = s_b.solve(0.0, 0.0, &options).expect("parallel solve b");
    for j in 0..k {
        assert_eq!(
            db_a[j].to_bits(),
            db_b[j].to_bits(),
            "streaming Δβ must be deterministic run-to-run at {j}"
        );
    }
    for i in 0..dt_a.len() {
        assert_eq!(
            dt_a[i].to_bits(),
            dt_b[i].to_bits(),
            "streaming Δt must be deterministic run-to-run at {i}"
        );
    }

    // (b) accumulate_chunk parallel-vs-serial equivalence. A single big chunk
    // (>= MIN) takes the rayon fold; many tiny chunks (each < MIN) take the
    // in-place serial path. Same row contributions, so the reduced Schur block
    // and reduced RHS agree to ULP-scale reassociation error.
    let mut par = StreamingArrowSchur::from_system(&sys, n);
    par.reset_accumulator(0.0).expect("reset par");
    par.accumulate_chunk(0, n, 0.0, ArrowSolverMode::Direct)
        .expect("parallel accumulate");
    let (s_par, rhs_par) = par.take_accumulators();

    let mut ser = StreamingArrowSchur::from_system(&sys, 8);
    ser.reset_accumulator(0.0).expect("reset ser");
    for start in (0..n).step_by(8) {
        let end = (start + 8).min(n);
        assert!(end - start < SCHUR_MATVEC_PARALLEL_ROW_MIN); // serial per chunk
        ser.accumulate_chunk(start, end, 0.0, ArrowSolverMode::Direct)
            .expect("serial accumulate");
    }
    let (s_ser, rhs_ser) = ser.take_accumulators();

    let s_scale = s_ser.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
    let mut s_max = 0.0_f64;
    for (a, b) in s_par.iter().zip(s_ser.iter()) {
        s_max = s_max.max((a - b).abs());
    }
    assert!(
        s_max / s_scale < 1e-12,
        "parallel vs serial reduced-Schur block diverges by rel {:e}",
        s_max / s_scale
    );
    let r_scale = rhs_ser
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    let mut r_max = 0.0_f64;
    for (a, b) in rhs_par.iter().zip(rhs_ser.iter()) {
        r_max = r_max.max((a - b).abs());
    }
    assert!(
        r_max / r_scale < 1e-12,
        "parallel vs serial reduced-RHS diverges by rel {:e}",
        r_max / r_scale
    );

    // (c) back_substitute parallel-vs-sequential: per-row writes are disjoint,
    // so the parallel scatter must match the hand-rolled sequential back-solve
    // BIT-FOR-BIT (no reassociation — each segment is computed identically).
    let s_bs = StreamingArrowSchur::from_system(&sys, n);
    // Use the already-solved Δβ as the back-substitution input.
    let delta_t = s_bs
        .back_substitute(0.0, db_a.view())
        .expect("parallel back_substitute");
    // Sequential reference: replicate the per-row formula directly.
    let backend = CpuBatchedBlockSolver;
    let total_len: usize = (0..n).map(|i| sys.rows[i].htt.nrows()).sum();
    let mut ref_dt = Array1::<f64>::zeros(total_len);
    let mut base = 0usize;
    for i in 0..n {
        let row = &sys.rows[i];
        let di = row.htt.nrows();
        let factor = factor_one_row(row, 0.0, di, i, false).expect("factor row");
        let mut rhs = Array1::<f64>::zeros(di);
        for c in 0..di {
            let mut acc = 0.0_f64;
            for a in 0..k {
                acc += row.htbeta[[c, a]] * db_a[a];
            }
            rhs[c] = row.gt[c] + acc;
        }
        let dt_i = backend.solve_block_vector(factor.view(), rhs.view());
        for c in 0..di {
            ref_dt[base + c] = -dt_i[c];
        }
        base += di;
    }
    for i in 0..total_len {
        assert_eq!(
            delta_t[i].to_bits(),
            ref_dt[i].to_bits(),
            "parallel back_substitute must match sequential bit-for-bit at {i}"
        );
    }
}

/// #1017 streaming-assembly speedup bench. Times the reduced-Schur + reduced-RHS
/// assembly (`accumulate_chunk`) at the SAE-arm shape, serial (tiny sub-MIN
/// chunks) vs parallel (one big chunk over rayon). Run with `--release
/// --nocapture` on a quiet multicore box to read the wall-clock and speedup;
/// the assembly is paid once per outer evaluation in the streaming joint fit.
///
/// ```text
/// cargo test --lib --release \
///   solver::arrow_schur::tests::bench_streaming_assembly_parallel_speedup \
///   -- --nocapture
/// ```
#[test]
pub(crate) fn bench_streaming_assembly_parallel_speedup() {
    let n = 1500usize;
    let d = 6usize;
    let k = 512usize;
    let sys = dense_direct_system(n, d, k);
    let calls = 10usize;
    let mut sink = 0.0_f64;

    // Serial: tiny chunks (each < SCHUR_MATVEC_PARALLEL_ROW_MIN) take the
    // in-place per-row path. Warm once before timing.
    let serial_assemble = || -> f64 {
        let mut s = StreamingArrowSchur::from_system(&sys, 8);
        s.reset_accumulator(0.0).expect("reset");
        for start in (0..n).step_by(8) {
            let end = (start + 8).min(n);
            s.accumulate_chunk(start, end, 0.0, ArrowSolverMode::Direct)
                .expect("serial accumulate");
        }
        let (s_acc, _) = s.take_accumulators();
        s_acc[[0, 0]]
    };
    sink += serial_assemble();
    let t_seq = std::time::Instant::now();
    for _ in 0..calls {
        sink += serial_assemble();
    }
    let seq_per = t_seq.elapsed().as_secs_f64() / calls as f64;

    // Parallel: one big chunk (>= MIN) fans over rayon. Warm once.
    let par_assemble = || -> f64 {
        let mut s = StreamingArrowSchur::from_system(&sys, n);
        s.reset_accumulator(0.0).expect("reset");
        s.accumulate_chunk(0, n, 0.0, ArrowSolverMode::Direct)
            .expect("parallel accumulate");
        let (s_acc, _) = s.take_accumulators();
        s_acc[[0, 0]]
    };
    sink += par_assemble();
    let t_par = std::time::Instant::now();
    for _ in 0..calls {
        sink += par_assemble();
    }
    let par_per = t_par.elapsed().as_secs_f64() / calls as f64;

    println!(
        "[#1017 streaming assembly, n={n} d={d} k={k}, {calls} calls, \
             {} rayon threads]\n  serial:   {:.3} ms/call\n  parallel: {:.3} ms/call\n  \
             speedup:  {:.2}x  (sink {:.3e})",
        rayon::current_num_threads(),
        seq_per * 1e3,
        par_per * 1e3,
        seq_per / par_per,
        sink,
    );
    assert!(seq_per > 0.0 && par_per > 0.0, "timings must be positive");
}

/// #1017 preconditioner-build parallelism: `JacobiPreconditioner::build_block_jacobi`
/// — the term-block-Jacobi PCG preconditioner built once per inexact-PCG solve
/// (so O(inner-Newton-iters) times per fit) — fans its per-row reduced-Schur
/// sub-block sweep over rayon above `SCHUR_MATVEC_PARALLEL_ROW_MIN`. It must be
/// (a) DETERMINISTIC run-to-run — bit-identical regardless of thread scheduling
/// (the preconditioner, hence the criterion ranking, cannot move); and
/// (b) numerically equal to the sequential per-row fold up to ULP-level
/// reassociation. Asserted through the applied output `P⁻¹ r` (the factored
/// block apply), which is what the PCG iterate actually consumes.
#[test]
pub(crate) fn parallel_block_jacobi_deterministic_and_matches_sequential() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64; // trips the parallel path
    let d = 4usize;
    let k = 24usize;
    let mut sys = dense_direct_system(n, d, k);
    // Partition the border into 4 blocks of 6 (each < BLOCK_JACOBI_MAX_BLOCK),
    // so `build_block_jacobi` is the path taken.
    let offsets: Vec<std::ops::Range<usize>> = (0..k).step_by(6).map(|s| s..(s + 6)).collect();
    sys.set_block_offsets(offsets.into());
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;
    let r = Array1::from_iter((0..k).map(|a| 0.4 * ((a as f64) * 0.019).cos() - 0.05));

    // (a) Determinism: two independent builds of the live (parallel) path must
    // apply bit-identically.
    let p_a = JacobiPreconditioner::build_block_jacobi(&sys, &htt_factors, ridge_beta, &backend)
        .expect("block Jacobi build a");
    let p_b = JacobiPreconditioner::build_block_jacobi(&sys, &htt_factors, ridge_beta, &backend)
        .expect("block Jacobi build b");
    let out_a = p_a.apply(&r);
    let out_b = p_b.apply(&r);
    for a in 0..k {
        assert_eq!(
            out_a[a].to_bits(),
            out_b[a].to_bits(),
            "parallel block Jacobi must apply deterministically at {a}"
        );
    }

    // (b) Equivalence with a hand-rolled sequential per-row reduced-Schur build.
    // Seed each block with H_ββ block-diag + ridge (here hbb is diagonal 6.0),
    // then subtract Σ_i H_βt^(i)(H_tt^(i))⁻¹H_tβ^(i) row by row.
    let mut ref_blocks: Vec<Array2<f64>> = Vec::new();
    for range in sys.block_offsets.iter() {
        let b = range.end - range.start;
        let mut blk = Array2::<f64>::zeros((b, b));
        for bi in 0..b {
            blk[[bi, bi]] = sys.hbb[[range.start + bi, range.start + bi]] + ridge_beta;
        }
        ref_blocks.push(blk);
    }
    for i in 0..n {
        let row = &sys.rows[i];
        let di = row.htt.nrows();
        let factor = factor_one_row(row, 0.0, di, i, false).expect("factor row");
        for (bidx, range) in sys.block_offsets.iter().enumerate() {
            let b = range.end - range.start;
            let mut solved_cols = Array2::<f64>::zeros((di, b));
            for bj in 0..b {
                let gj = range.start + bj;
                let rhs = row.htbeta.column(gj).to_owned();
                let solved = backend.solve_block_vector(factor.view(), rhs.view());
                for c in 0..di {
                    solved_cols[[c, bj]] = solved[c];
                }
            }
            for bi in 0..b {
                let gi = range.start + bi;
                for bj in 0..b {
                    let mut acc = 0.0;
                    for c in 0..di {
                        acc += row.htbeta[[c, gi]] * solved_cols[[c, bj]];
                    }
                    ref_blocks[bidx][[bi, bj]] -= acc;
                }
            }
        }
    }
    // Apply the reference block-diagonal inverse to r by Cholesky-solving each
    // assembled block (the same factor+solve `build_block_jacobi.apply` uses).
    let mut ref_out = Array1::<f64>::zeros(k);
    for (bidx, range) in sys.block_offsets.iter().enumerate() {
        let b = range.end - range.start;
        let llt = {
            use faer::Side;
            let view = crate::linalg::faer_ndarray::FaerArrayView::new(&ref_blocks[bidx]);
            crate::linalg::faer_ndarray::FaerLlt::new(view.as_ref(), Side::Lower)
                .expect("ref block must be PD")
        };
        let rhs = Array1::from_iter((0..b).map(|bi| r[range.start + bi]));
        use faer::linalg::solvers::Solve;
        let stride = rhs.strides()[0];
        let len = rhs.len();
        // SAFETY: `rhs` is a live `Array1<f64>` that outlives `rhs_mat` (both
        // dropped at the end of this loop iteration); `rhs.as_ptr()` is valid for
        // `len = rhs.len()` contiguous f64 reads, and the `(len, 1)` shape with
        // row stride `rhs.strides()[0]` and col stride 0 exactly describes that
        // single-column layout. No aliasing: the view is read-only and `rhs` is
        // not mutated while `rhs_mat` is borrowed.
        let rhs_mat = unsafe { faer::MatRef::from_raw_parts(rhs.as_ptr(), len, 1, stride, 0) };
        let solved = llt.solve(rhs_mat);
        for bi in 0..b {
            ref_out[range.start + bi] = solved[(bi, 0)];
        }
    }
    let scale = ref_out.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
    let mut max_abs = 0.0_f64;
    for a in 0..k {
        max_abs = max_abs.max((out_a[a] - ref_out[a]).abs());
    }
    assert!(
        max_abs / scale < 1e-10,
        "parallel block Jacobi apply diverges from sequential by rel {:e}",
        max_abs / scale
    );
}

/// #1017 block-Jacobi preconditioner-build speedup bench. Times
/// `build_block_jacobi` at the SAE-arm shape, sequential (forced via an inside-
/// worker call so the gate stays serial) vs the live parallel build. Run with
/// `--release --nocapture` on a quiet multicore box; the preconditioner is built
/// once per inexact-PCG solve in the streaming joint fit.
///
/// ```text
/// cargo test --lib --release \
///   solver::arrow_schur::tests::bench_block_jacobi_parallel_speedup -- --nocapture
/// ```
#[test]
pub(crate) fn bench_block_jacobi_parallel_speedup() {
    let n = 1500usize;
    let d = 6usize;
    let k = 480usize;
    let mut sys = dense_direct_system(n, d, k);
    // 80 blocks of 6 (< BLOCK_JACOBI_MAX_BLOCK) → the block-Jacobi path.
    let offsets: Vec<std::ops::Range<usize>> = (0..k).step_by(6).map(|s| s..(s + 6)).collect();
    sys.set_block_offsets(offsets.into());
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;
    let calls = 10usize;
    let mut sink = 0.0_f64;

    // Sequential baseline: the build's row gate stays serial when called from
    // inside a rayon worker (current_thread_index().is_some()).
    let seq_build = || -> f64 {
        std::iter::once(()).map(|_| {
            let p = JacobiPreconditioner::build_block_jacobi(&sys, &htt_factors, ridge_beta, &backend)
                .expect("serial block Jacobi");
            p.apply(&Array1::<f64>::ones(k))[0]
        }).sum::<f64>()
    };
    sink += seq_build();
    let t_seq = std::time::Instant::now();
    for _ in 0..calls {
        sink += seq_build();
    }
    let seq_per = t_seq.elapsed().as_secs_f64() / calls as f64;

    // Parallel: top-level call (not nested) trips the rayon path.
    let par_build = || -> f64 {
        let p = JacobiPreconditioner::build_block_jacobi(&sys, &htt_factors, ridge_beta, &backend)
            .expect("parallel block Jacobi");
        p.apply(&Array1::<f64>::ones(k))[0]
    };
    sink += par_build();
    let t_par = std::time::Instant::now();
    for _ in 0..calls {
        sink += par_build();
    }
    let par_per = t_par.elapsed().as_secs_f64() / calls as f64;

    println!(
        "[#1017 block-Jacobi build, n={n} d={d} k={k} ({} blocks), {calls} calls, \
             {} rayon threads]\n  serial:   {:.3} ms/call\n  parallel: {:.3} ms/call\n  \
             speedup:  {:.2}x  (sink {:.3e})",
        sys.block_offsets.len(),
        rayon::current_num_threads(),
        seq_per * 1e3,
        par_per * 1e3,
        seq_per / par_per,
        sink,
    );
    assert!(seq_per > 0.0 && par_per > 0.0, "timings must be positive");
}

/// #1017 scalar-Jacobi build parallelism: `build_scalar_jacobi` (the scalar-
/// diagonal PCG preconditioner taken for wide/absent block structure with no
/// SAE residency) fans its per-row diagonal sweep over rayon above
/// `SCHUR_MATVEC_PARALLEL_ROW_MIN`. Must be DETERMINISTIC run-to-run (bit-
/// identical apply). Numeric equivalence vs the resident path is already covered
/// by `resident_scalar_jacobi_matches_generic`; this pins run-to-run stability.
#[test]
pub(crate) fn parallel_scalar_jacobi_deterministic() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64;
    let d = 4usize;
    let k = 24usize;
    let sys = dense_direct_system(n, d, k); // no block_offsets, no resident → scalar path
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;
    let r = Array1::from_iter((0..k).map(|a| 0.3 * ((a as f64) * 0.023).sin() + 0.11));

    let p_a = JacobiPreconditioner::build_scalar_jacobi(&sys, &htt_factors, ridge_beta, &backend)
        .expect("scalar Jacobi a");
    let p_b = JacobiPreconditioner::build_scalar_jacobi(&sys, &htt_factors, ridge_beta, &backend)
        .expect("scalar Jacobi b");
    let out_a = p_a.apply(&r);
    let out_b = p_b.apply(&r);
    for a in 0..k {
        assert_eq!(
            out_a[a].to_bits(),
            out_b[a].to_bits(),
            "parallel scalar Jacobi must apply deterministically at {a}"
        );
    }
}

/// #1017 scalar-Jacobi build speedup bench (serial via nested-worker gate vs the
/// live parallel build). Run with `--release --nocapture`.
#[test]
pub(crate) fn bench_scalar_jacobi_parallel_speedup() {
    let n = 1500usize;
    let d = 6usize;
    let k = 480usize;
    let sys = dense_direct_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;
    let calls = 10usize;
    let mut sink = 0.0_f64;

    let seq_build = || -> f64 {
        rayon::iter::once(())
            .map(|_| {
                JacobiPreconditioner::build_scalar_jacobi(&sys, &htt_factors, ridge_beta, &backend)
                    .expect("serial scalar Jacobi")
                    .apply(&Array1::<f64>::ones(k))[0]
            })
            .sum::<f64>()
    };
    sink += seq_build();
    let t_seq = std::time::Instant::now();
    for _ in 0..calls {
        sink += seq_build();
    }
    let seq_per = t_seq.elapsed().as_secs_f64() / calls as f64;

    let par_build = || -> f64 {
        JacobiPreconditioner::build_scalar_jacobi(&sys, &htt_factors, ridge_beta, &backend)
            .expect("parallel scalar Jacobi")
            .apply(&Array1::<f64>::ones(k))[0]
    };
    sink += par_build();
    let t_par = std::time::Instant::now();
    for _ in 0..calls {
        sink += par_build();
    }
    let par_per = t_par.elapsed().as_secs_f64() / calls as f64;

    println!(
        "[#1017 scalar-Jacobi build, n={n} d={d} k={k}, {calls} calls, \
             {} rayon threads]\n  serial:   {:.3} ms/call\n  parallel: {:.3} ms/call\n  \
             speedup:  {:.2}x  (sink {:.3e})",
        rayon::current_num_threads(),
        seq_per * 1e3,
        par_per * 1e3,
        seq_per / par_per,
        sink,
    );
    assert!(seq_per > 0.0 && par_per > 0.0, "timings must be positive");
}

/// #1017 `arrow_operator_infinity_norm` must equal the brute-force inf-norm of
/// the fully-assembled arrow operator `[[H_tt+ρ_t I, H_tβ],[H_βt, H_ββ+ρ_β I]]`.
/// The optimized single-pass form (materialize each row's cross-block ONCE,
/// fold its column-abs into a length-K vector) replaced an `O(K·n·K²)`
/// re-materialization; it computes the SAME absolute row sums, so it must match
/// a dense assembly bit-for-bit (same terms, same per-column accumulation order).
#[test]
pub(crate) fn arrow_operator_infinity_norm_matches_dense_assembly() {
    let n = 12usize;
    let d = 3usize;
    let k = 7usize;
    let sys = dense_direct_system(n, d, k);
    let ridge_t = 0.3_f64;
    let ridge_beta = 0.2_f64;

    let got = arrow_operator_infinity_norm(&sys, ridge_t, ridge_beta).expect("inf-norm");

    // Brute-force dense assembly: total dim = n*d (t) + k (beta).
    let total = n * d + k;
    let mut full = Array2::<f64>::zeros((total, total));
    let hbb = sys.effective_penalty_op().to_dense();
    // t-blocks on the diagonal + cross-blocks H_tβ / H_βt.
    for i in 0..n {
        let base = i * d;
        let row = &sys.rows[i];
        let htbeta = sys_htbeta_materialize_row(&sys, i, row).expect("materialize");
        for a in 0..d {
            for b in 0..d {
                full[[base + a, base + b]] = row.htt[[a, b]];
            }
            full[[base + a, base + a]] += ridge_t;
            for bc in 0..k {
                let v = htbeta[[a, bc]];
                full[[base + a, n * d + bc]] = v; // H_tβ
                full[[n * d + bc, base + a]] = v; // H_βt (symmetric)
            }
        }
    }
    for br in 0..k {
        for bc in 0..k {
            full[[n * d + br, n * d + bc]] += hbb[[br, bc]];
        }
        full[[n * d + br, n * d + br]] += ridge_beta;
    }
    let mut want = 0.0_f64;
    for r in 0..total {
        let mut s = 0.0_f64;
        for c in 0..total {
            s += full[[r, c]].abs();
        }
        want = want.max(s);
    }
    let scale = want.max(1.0);
    assert!(
        (got - want).abs() / scale < 1e-12,
        "arrow inf-norm {got} != dense assembly {want} (rel {:e})",
        (got - want).abs() / scale
    );
}
