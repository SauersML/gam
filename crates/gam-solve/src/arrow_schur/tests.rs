//! Unit tests for the arrow-Schur solver.
#![cfg(test)]

use super::*;
use approx::assert_abs_diff_eq;
use ndarray::array;

fn gpu_available_or_fail() -> bool {
    gam_gpu::device_runtime::GpuRuntime::resolve(gam_gpu::GpuPolicy::Auto)
        .unwrap_or_else(|error| panic!("GPU probe fault in Arrow-Schur test: {error}"))
        .is_some()
}

#[test]
fn arrow_solve_options_own_gpu_policy_2322() {
    let off = ArrowSolveOptions::direct().with_gpu_policy(gam_gpu::GpuPolicy::Off);
    let required = ArrowSolveOptions::direct().with_gpu_policy(gam_gpu::GpuPolicy::Required);
    assert_eq!(off.gpu_policy, gam_gpu::GpuPolicy::Off);
    assert_eq!(required.gpu_policy, gam_gpu::GpuPolicy::Required);
}

/// #1995: compact SAE rows hand `block_gemm_subtract` dense scratch matrices
/// whose nonzeros occupy only the active top-k beta columns. The CPU fallback
/// must produce the same Schur update as a dense GEMM while doing work only on
/// the discovered column support.
#[test]
pub(crate) fn block_gemm_subtract_matches_dense_on_sparse_column_support() {
    let backend = CpuBatchedBlockSolver;
    let d = 3usize;
    let k = 12usize;
    let mut left = Array2::<f64>::zeros((d, k));
    let mut right = Array2::<f64>::zeros((d, k));
    for (row, col, value) in [
        (0, 1, 0.7),
        (1, 1, -0.2),
        (2, 7, 1.3),
        (0, 10, -0.4),
        (2, 10, 0.9),
    ] {
        left[[row, col]] = value;
    }
    for (row, col, value) in [
        (0, 2, -1.1),
        (2, 2, 0.5),
        (1, 7, 0.8),
        (0, 11, 0.25),
        (2, 11, -0.6),
    ] {
        right[[row, col]] = value;
    }

    let mut actual = Array2::<f64>::zeros((k, k));
    backend.block_gemm_subtract(&mut actual, &left, &right);

    let mut expected = Array2::<f64>::zeros((k, k));
    for c in 0..d {
        for a in 0..k {
            for b in 0..k {
                expected[[a, b]] -= left[[c, a]] * right[[c, b]];
            }
        }
    }
    for a in 0..k {
        for b in 0..k {
            assert_eq!(actual[[a, b]], expected[[a, b]], "entry ({a}, {b})");
        }
    }
}

fn beta_gauge_evidence_fixture(gauge_row: [f64; 3]) -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(0, 0, 3);
    sys.hbb = array![
        [gauge_row[0], gauge_row[1], gauge_row[2]],
        [gauge_row[1], 4.0, 1.0],
        [gauge_row[2], 1.0, 5.0]
    ];
    sys.gb = array![-13.0, -2.0, 1.0];
    sys.set_beta_gauge_quotient(
        ArrowBetaGaugeQuotient::new(vec![array![1.0, 0.0, 0.0]]).expect("gauge"),
    )
    .expect("matching border");
    sys
}

/// #2228 — the wide-`p` InexactPCG Newton step must gauge-fix identically to the
/// dense Faddeev–Popov pin. Forcing the matrix-free lane on a gauge-quotiented
/// system, the step must carry no gauge-orbit component (`Q·Δβ ≈ 0`) and match
/// the dense Direct-mode pinned step componentwise on the identifiable
/// complement. Before the fix this call `Err`ed ("InexactPCG does not return an
/// evidence factor"); an un-pinned PCG solve would instead leave an arbitrary
/// component along the singular gauge direction `[1, 0, 0]`.
#[test]
pub(crate) fn inexact_pcg_gauge_quotient_projects_step_and_matches_dense_pin_2228() {
    let sys = beta_gauge_evidence_fixture([7.0, 2.0, -3.0]);
    let (_, dbeta_direct, _) = solve_arrow_newton_step_with_options(
        &sys,
        0.0,
        0.0,
        &ArrowSolveOptions::direct().with_positive_definite_evidence(),
    )
    .expect("dense pinned step");
    // Force the matrix-free lane on the SAME small gauge-quotiented system, with
    // a tight CG tolerance so the componentwise match is not limited by the loose
    // default ranking tolerance.
    let mut pcg_options = ArrowSolveOptions::inexact_pcg().with_positive_definite_evidence();
    pcg_options.pcg.relative_tolerance = 1e-12;
    pcg_options.trust_region.steihaug_relative_tolerance = 1e-12;
    let (_, dbeta_pcg, _) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &pcg_options)
        .expect("matrix-free pinned step");
    // (a) no gauge-orbit motion: the declared gauge direction `[1, 0, 0]` is
    // erased from the matrix-free step, exactly as the dense pin erases it.
    let gauge = array![1.0, 0.0, 0.0];
    assert_abs_diff_eq!(gauge.dot(&dbeta_pcg), 0.0, epsilon = 1e-11);
    // (b) the matrix-free step matches the dense Faddeev–Popov step.
    for i in 0..3 {
        assert_abs_diff_eq!(dbeta_pcg[i], dbeta_direct[i], epsilon = 1e-9);
    }
}

/// #2228 byte-exact guard: routing the fit-step matvec through
/// `ReducedSchurOperator` is bit-identical to the bare `schur_matvec` apply when
/// the system carries no β-gauge quotient, so the wide-`p` InexactPCG lane is
/// unchanged for every non-SAE-fit caller.
#[test]
pub(crate) fn reduced_schur_operator_matvec_is_bit_identical_without_quotient_2228() {
    let mut sys = ArrowSchurSystem::new(0, 0, 3);
    sys.hbb = array![[6.0, 1.0, -2.0], [1.0, 4.0, 0.5], [-2.0, 0.5, 5.0]];
    assert!(sys.beta_gauge_quotient.is_none());
    let factors = ArrowFactorSlab::from_blocks(Vec::new());
    let backend = CpuBatchedBlockSolver;
    let x = array![0.7, -1.3, 2.1];
    let routed = ReducedSchurOperator::new(&sys, &factors, 0.0, &backend, None).apply_owned(&x);
    let mut bare = array![0.0, 0.0, 0.0];
    schur_matvec(&sys, &factors, 0.0, &x, &mut bare, &backend, None);
    for i in 0..3 {
        assert_eq!(routed[i].to_bits(), bare[i].to_bits(), "matvec entry {i}");
    }
}

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
            |delta_t, _| quartic_counterexample_value(t + delta_t[0]),
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

    // Evidence/log-det factorization must
    // accept the same barely-PD block and return its genuine Cholesky
    // factor — the diagonal gives an exact log-determinant.
    let factor = factor_one_row(&row, 0.0, d, 0, true)
        .expect("evidence factorization must accept a barely-PD-but-PD block");
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
        "non-PD block must error without an explicit deflation policy; got {npd:?}"
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
/// off the gauge orbit (the K>1 ordered-Beta--Bernoulli/softmax row-sharing state) must be
/// conditioned by the undamped evidence factor through **unit-stiffness
/// spectral deflation** — `factor_spectral_deflated_criterion_row_with_geometry` discovers
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
    let spectral = factor_spectral_deflated_criterion_row_with_geometry(&indef, d, false, None)
        .expect("the majorizer policy never refuses on sign")
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

    // The undamped evidence factor (evidence policy, ridge_t = 0,
    // gauge passed in) now SUCCEEDS on this block via spectral deflation
    // rather than refusing — so the SAE driver gets a finite, BIAS-FREE
    // evidence cache and never falls back to a ρ-dependent ridge.
    let factored = factor_one_row_result(
        &indef,
        0.0,
        d,
        0,
        true,
        std::slice::from_ref(&gauge_e1),
        true,
        false,
        None,
    )
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

    let result =
        factor_one_row_result(&pd, 0.0, d, 0, true, std::slice::from_ref(&gauge_e1), true, false, None)
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

/// #1273 regression — the SAE evidence path must recover a per-row `H_tt`
/// that is rank-deficient because the atom's data is intrinsically LOWER-
/// dimensional than its chart (the reported circle/torus case: a 1-D ring
/// embedded in a 2-D torus harmonic basis), even when THIS row carries NO
/// supplied gauge direction that spans the flat direction.
///
/// This block has a genuine FLAT tangent direction (a numerically-zero
/// eigenvalue along e_1) but is otherwise PD and finite — the REML cost is
/// valid; the per-row tangent Hessian simply has a null direction from the
/// intrinsic-dimension deficiency, NOT a broken/NaN state. Before the fix the
/// undamped evidence factor's spectral discovery-deflation was gated behind
/// `!row_gauges.is_empty()`, so a row whose flat direction was intrinsic-
/// dimension deficiency (not a supplied rotation/phase gauge) hit the hard
/// "H_tt is non-PD at base ridge" refusal — which the SAE driver surfaced all
/// the way out as the issue's `RemlConvergenceError`. After the fix the SAE
/// evidence path (`allow_spectral_deflation = true`) DISCOVERS the flat
/// direction from the block's own eigendecomposition and unit-stiffness
/// deflates it (a ρ-independent `log 1 = 0`), so the factorization SUCCEEDS
/// with no gauge supplied and no ρ-dependent ridge bias.
#[test]
pub(crate) fn evidence_row_recovers_intrinsic_dimension_flat_block_without_gauge_1273() {
    let d = 2usize; // d_atom = 2 chart.
    let k = 1usize; // K = 1 atom.

    // A 2-D chart over 1-D ring data: the tangent Hessian is PD along the
    // ring direction (e_0, curvature 3.0) and FLAT along the ambient
    // direction the data never explores (e_1, curvature exactly 0). This is
    // the genuine rank-1 deficiency `H_tt` carries on the #1273 geometry; it
    // is finite and not indefinite, so the REML cost at this ρ is valid — the
    // factorization must NOT abort.
    let mut flat = ArrowRowBlock::new(d, k);
    flat.htt = array![[3.0_f64, 0.0], [0.0, 0.0]];
    flat.htbeta = array![[1.0_f64], [0.5]];
    flat.gt = array![0.0_f64, 0.0];

    // Precondition: the undamped (ridge_t = 0) Cholesky genuinely REFUSES the
    // flat block — without this the factorization would just succeed and the
    // fix would not be exercised. With NO supplied gauge AND spectral
    // deflation withheld (the pre-#1273 behaviour the empty-gauge gate forced
    // on this row), the block is rejected as non-PD.
    let refused = factor_one_row_result(&flat, 0.0, d, 0, true, &[], false, false, None);
    assert!(
        refused.is_err(),
        "fixture precondition: the rank-deficient flat H_tt must be refused by \
         the undamped evidence factor when spectral deflation is withheld and no \
         gauge is supplied — the exact pre-#1273 abort"
    );

    // The fix: the SAE evidence path opts into spectral discovery-deflation,
    // which finds the flat e_1 direction from the block's own
    // eigendecomposition and stiffens it to unit curvature, producing an SPD
    // factor — so the factorization SUCCEEDS with no gauge supplied and the
    // #1273 fit no longer aborts on this legitimately-flat geometry.
    let recovered = factor_one_row_result(&flat, 0.0, d, 0, true, &[], true, false, None).expect(
        "spectral deflation must recover the intrinsic-dimension flat H_tt block on \
         the SAE evidence path even with no supplied gauge (#1273)",
    );
    assert_eq!(
        recovered.gauge_deflated_directions, 1,
        "exactly the single intrinsic-dimension flat direction must be deflated"
    );
    // The factor must be a valid SPD Cholesky (finite, positive pivots).
    for a in 0..d {
        assert!(
            recovered.factor[[a, a]].is_finite() && recovered.factor[[a, a]] > 0.0,
            "recovered evidence factor must have a finite positive pivot at {a}; got {}",
            recovered.factor[[a, a]]
        );
    }
    // The genuine ring direction e_0 is preserved exactly (no ridge bias); the
    // deflated flat direction e_1 carries exactly the +1 unit stiffness
    // (`log 1 = 0`, ρ-independent), NOT a magic ridge constant.
    let l = &recovered.factor;
    let mut recon = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0_f64;
            for kk in 0..d {
                acc += l[[i, kk]] * l[[j, kk]];
            }
            recon[[i, j]] = acc;
        }
    }
    assert!(
        (recon[[0, 0]] - 3.0).abs() < 1.0e-12,
        "genuine ring direction e_0 must be preserved exactly; got {}",
        recon[[0, 0]]
    );
    assert!(
        (recon[[1, 1]] - 1.0).abs() < 1.0e-9,
        "the intrinsic-dimension flat direction e_1 must be unit-stiffness \
         deflated to exactly +1 (log 1 = 0, ρ-independent), NOT ridge-damped; got {}",
        recon[[1, 1]]
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
/// (`record_criterion_gauge_deflation_count`) mid-optimization — the slow
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

    let lo = factor_spectral_deflated_criterion_row_with_geometry(&block_lo, d, false, None)
        .expect("the majorizer policy never refuses on sign (lo iterate)")
        .expect("indefinite block must spectrally deflate (lo iterate)");
    let hi = factor_spectral_deflated_criterion_row_with_geometry(&block_hi, d, false, None)
        .expect("the majorizer policy never refuses on sign (hi iterate)")
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
    let lo_conditioning = &lo
        .deflation_spectrum
        .as_ref()
        .expect("lo iterate spectral metadata")
        .conditioning;
    let hi_conditioning = &hi
        .deflation_spectrum
        .as_ref()
        .expect("hi iterate spectral metadata")
        .conditioning;
    assert_eq!(
        &**lo_conditioning,
        &[
            RowSpectralConditioning::UnitDeflated,
            RowSpectralConditioning::FloorClamped,
            RowSpectralConditioning::Raw,
        ],
        "the lo iterate must expose the classifier's floor-clamped branch"
    );
    assert_eq!(
        &**hi_conditioning,
        &[
            RowSpectralConditioning::UnitDeflated,
            RowSpectralConditioning::Raw,
            RowSpectralConditioning::Raw,
        ],
        "the hi iterate must expose the classifier's raw-retained branch"
    );
    assert_ne!(
        lo_conditioning, hi_conditioning,
        "equal deflation counts must not hide a floor-clamp stratum crossing"
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

/// #2572 — the β-coupling graph must be identical whether `H_tβ` arrives as a
/// dense per-row slab or as the routed matvec pair, and a matrix-free system
/// (whose row slabs are allocated at ZERO columns) must not be subscripted.
///
/// Before the fix this built from `sys.rows[i].htbeta` directly. On the
/// overcomplete support-sparse lane — `htbeta_cols = 0` plus per-atom
/// `block_offsets` — that read `(d, 0)[[axis, col]]` and aborted with
/// `ndarray: index out of bounds`, on whichever rayon worker the escalated PCG
/// tier ran on. Measured on a seeded `K = 24 > P = 8`, `top_k = 4` term:
/// `ClusterJacobiPreconditioner::from_arrow_schur` and
/// `AdditiveSchwarzPreconditioner::from_arrow_schur` both aborted with exactly
/// that message (`examples/issue_2572_precond_probe.rs` in `gam-sae`).
#[test]
fn beta_coupling_graph_reads_the_routed_htbeta_not_the_dense_slab() {
    // Four β blocks of two columns each. Rows 0 and 1 co-fire blocks (0, 1);
    // row 2 co-fires (2, 3); nothing bridges the two pairs, so the component
    // partition must be exactly {{0, 1}, {2, 3}} and block 0 must never be
    // reported as co-firing with block 2.
    let k = 8usize;
    let block_offsets: Arc<[Range<usize>]> = vec![0..2, 2..4, 4..6, 6..8].into();
    let entries: [(usize, [(usize, f64); 4]); 3] = [
        (0, [(0, 1.0), (1, -0.5), (2, 0.25), (3, 2.0)]),
        (1, [(0, 0.5), (1, 1.5), (2, -1.0), (3, 0.75)]),
        (2, [(4, 1.0), (5, -2.0), (6, 0.5), (7, 1.25)]),
    ];

    let dense = {
        let mut sys = ArrowSchurSystem::new(3, 1, k);
        for (row, cols) in entries {
            for (col, value) in cols {
                sys.rows[row].htbeta[[0, col]] = value;
            }
        }
        sys.set_block_offsets(Arc::clone(&block_offsets));
        sys
    };
    let matrix_free = {
        // Same operator, no dense slab at all: `htbeta_cols = 0`, exactly what
        // `SaeSupportSparseTerm::assemble_arrow_schur` allocates.
        let mut sys = ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(
            vec![1, 1, 1],
            k,
            0,
        );
        assert_eq!(sys.rows[0].htbeta.dim(), (1, 0));
        sys.set_row_htbeta_operator(
            move |row_idx, x, out| {
                for (row, cols) in entries {
                    if row == row_idx {
                        for (col, value) in cols {
                            out[0] += value * x[col];
                        }
                    }
                }
            },
            move |row_idx, v, out| {
                for (row, cols) in entries {
                    if row == row_idx {
                        for (col, value) in cols {
                            out[col] += value * v[0];
                        }
                    }
                }
            },
        );
        sys.set_block_offsets(Arc::clone(&block_offsets));
        sys
    };

    let dense_graph = BetaCouplingGraph::build_from_system(&dense);
    let free_graph = BetaCouplingGraph::build_from_system(&matrix_free);
    let partition = |graph: &BetaCouplingGraph| -> Vec<Vec<usize>> {
        let mut parts = graph.component_partition();
        for part in parts.iter_mut() {
            part.sort_unstable();
        }
        parts.sort();
        parts
    };
    assert_eq!(partition(&dense_graph), vec![vec![0, 1], vec![2, 3]]);
    assert_eq!(
        partition(&free_graph),
        partition(&dense_graph),
        "the routed operator and the dense slab describe the same H_tbeta, so \
         they must describe the same coupling graph"
    );
    let weights = |graph: &BetaCouplingGraph| -> Vec<(usize, usize, f64)> {
        graph
            .edges
            .iter()
            .map(|edge| {
                let weight = graph
                    .weighted_neighbours(edge.a)
                    .find(|(node, _)| *node == edge.b)
                    .map(|(_, weight)| weight)
                    .expect("every edge carries its co-firing weight");
                (edge.a, edge.b, weight)
            })
            .collect()
    };
    assert_eq!(
        weights(&dense_graph),
        vec![(0, 1, 2.0), (2, 3, 1.0)],
        "rows 0 and 1 co-fire blocks (0, 1); row 2 co-fires (2, 3)"
    );
    assert_eq!(weights(&free_graph), weights(&dense_graph));

    // A column that is nonzero in two latent rows but sums to zero across them
    // is still ACTIVE: the predicate is "some entry is nonzero", which the
    // element scan tested and an unsigned probe would have missed.
    let cancelling = {
        let mut sys = ArrowSchurSystem::new_with_per_row_dims_empty_hbb_and_htbeta_cols(
            vec![2],
            4,
            0,
        );
        sys.set_row_htbeta_operator(
            |_, x, out| {
                out[0] += x[0] + x[2];
                out[1] += -x[0] + x[2];
            },
            |_, v, out| {
                out[0] += v[0] - v[1];
                out[2] += v[0] + v[1];
            },
        );
        sys.set_block_offsets(vec![0..2, 2..4].into());
        sys
    };
    let mut cancelling_parts = BetaCouplingGraph::build_from_system(&cancelling).component_partition();
    for part in cancelling_parts.iter_mut() {
        part.sort_unstable();
    }
    assert_eq!(
        cancelling_parts,
        vec![vec![0, 1]],
        "column 0 cancels under a ones-probe but is genuinely active, so both \
         blocks co-fire and the graph is one component"
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

#[test]
fn sparse_htbeta_transpose_never_probes_the_forward_operator() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let forward_calls = std::sync::Arc::new(AtomicUsize::new(0));
    let transpose_calls = std::sync::Arc::new(AtomicUsize::new(0));
    let mut sys = ArrowSchurSystem::new(1, 1, 3);
    let forward_counter = std::sync::Arc::clone(&forward_calls);
    let transpose_counter = std::sync::Arc::clone(&transpose_calls);
    sys.set_row_htbeta_operator(
        move |_, x, out| {
            forward_counter.fetch_add(1, Ordering::SeqCst);
            out[0] = 2.0 * x[0] - x[1] + 0.5 * x[2];
        },
        move |_, v, out| {
            transpose_counter.fetch_add(1, Ordering::SeqCst);
            out[0] += 2.0 * v[0];
            out[1] -= v[0];
            out[2] += 0.5 * v[0];
        },
    );

    let mut direct = Array1::<f64>::zeros(3);
    sys_htbeta_accumulate_transpose(&sys, 0, &sys.rows[0], array![3.0].view(), &mut direct);
    assert_eq!(direct, array![6.0, -3.0, 1.5]);
    assert_eq!(forward_calls.load(Ordering::SeqCst), 0);
    assert_eq!(transpose_calls.load(Ordering::SeqCst), 1);

    let cache = ArrowHtbetaCache::Matvec {
        op: std::sync::Arc::clone(sys.htbeta_matvec.as_ref().expect("forward")),
        transpose_op: std::sync::Arc::clone(
            sys.htbeta_transpose_matvec.as_ref().expect("transpose"),
        ),
        estimated_bytes: 3 * std::mem::size_of::<f64>(),
    };
    let mut cached = Array1::<f64>::zeros(3);
    assert!(cache.apply_row_transpose_accumulate(0, array![4.0].view(), &mut cached, 1, 3, None,));
    assert_eq!(cached, array![8.0, -4.0, 2.0]);
    assert_eq!(
        forward_calls.load(Ordering::SeqCst),
        0,
        "the O(K) forward-probe path must remain unreachable"
    );
    assert_eq!(transpose_calls.load(Ordering::SeqCst), 2);
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
    let dense_log_det: f64 = (0..l.nrows()).map(|i| 2.0 * l[[i, i]].ln()).sum();
    let cached_log_det = cache
        .joint_hessian_log_det
        .expect("direct undamped solve must cache the joint Hessian log-det");
    assert!(
        (cached_log_det - dense_log_det).abs() < 1.0e-9,
        "cached joint Hessian log-det {cached_log_det} vs dense {dense_log_det}"
    );
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
/// The positive-definite evidence policy accepts the RAW (undamped) blocks.
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
    let opts = ArrowSolveOptions::direct().with_positive_definite_evidence();
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
    let log_det_cache = cache.arrow_log_det().expect("authoritative joint logdet");

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
    let policy = gam_gpu::policy::GpuDispatchPolicy::default();
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
/// typed device-absence short-circuit makes the helper
/// itself return `None` on a CPU-only host, so the routing logic is asserted
/// through the predicate it consults (the device==CPU 1e-10 numeric parity is
/// asserted by the box harness).
#[test]
pub(crate) fn matvec_gate_engages_for_llm_shape_off_for_tiny() {
    let policy = gam_gpu::policy::GpuDispatchPolicy::default();
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

/// #1017 Phase-1 dispatch re-key (kernel side): the device matrix-free SAE
/// reduced-Schur PCG (`crate::gpu_kernels::arrow_schur::gpu_schur_matvec_backend`)
/// previously gated on the dense-Direct floor `dense_hessian_work_target_is_gpu(n,
/// k)`, the same floor `try_device_arrow_direct` (the single dense factorization)
/// uses. That is the wrong gate for the amortised matvec: it keys on `2·n·k²`,
/// dropping the per-row frame depth `d` (M) that multiplies the per-apply work and
/// the `1/cg_iters` staging amortisation. The kernel now consults the SAME
/// work-based predicate the host injection gate (`maybe_inject_gpu_schur_matvec`)
/// uses — `reduced_schur_matvec_should_offload(n, k, d, max_iterations)` — so the
/// two SAE-matvec dispatch sites cannot drift, and the gate registers the true
/// `n × k × d × cg_iters` batched work. This asserts that policy invariant on any
/// host (the predicates are pure; the device==CPU 1e-10 numeric parity stays the
/// box harness's job).
#[test]
pub(crate) fn matrix_free_sae_gate_uses_work_predicate_not_dense_floor() {
    let policy = gam_gpu::policy::GpuDispatchPolicy::default();
    // SAE matrix-free shape with a SMALL CG budget. The dense `(n, k)` floor the
    // kernel used to consult ignores both `d` and `cg_iters`, so at a thin border
    // and few iterations it can decline a shape whose true `n·k·d·cg_iters` work
    // clears the amortised breakeven. Pick a shape where keying on `d` matters:
    // wide-enough border to clear the device-loop floor, modest rows, real frame
    // depth.
    let (n, k, d) = (1_024_usize, 1_024_usize, 8_usize);
    let cg_iters = 8usize;
    // The re-keyed kernel admits this on the work predicate ...
    assert!(policy.reduced_schur_matvec_should_offload(n, k, d, cg_iters));
    // ... and stays off below the device-loop border floor regardless of how much
    // row/depth/iteration work piles up (launch latency per apply dominates).
    assert!(!policy.reduced_schur_matvec_should_offload(1_000_000, 16, 64, 64));
}

/// On a host without a CUDA device the production seam must decline (return
/// `None`), so `solve_arrow_newton_step_core` runs the unchanged CPU path
/// and the result equals the direct CPU artifacts solve bit-for-bit.
#[test]
pub(crate) fn device_seam_declines_without_gpu_and_matches_cpu() {
    if gpu_available_or_fail() {
        // On a CUDA host the device may legitimately serve the step; this
        // host-only invariant does not apply. The box harness asserts the
        // device==CPU 1e-10 parity instead.
        return;
    }
    let sys = dense_direct_system(6, 2, 4);
    let options = ArrowSolveOptions::direct();

    // The seam helpers both decline when no device is present.
    assert!(try_device_arrow_direct(&sys, 0.0, 0.0, &options).is_none());
    assert!(
        maybe_inject_gpu_schur_matvec(&sys, 0.0, 0.0, &options)
            .expect("GPU runtime resolution must not fault on the CPU host")
            .is_none()
    );

    // The public core entry therefore equals the direct CPU artifacts solve.
    let (dt_core, db_core, diag) =
        solve_arrow_newton_step_core(&sys, 0.0, 0.0, &options).expect("core solve");
    assert!(
        !diag.used_device_arrow,
        "no device present, so the solve must not be flagged device-served"
    );
    assert!(
        !diag.injected_host_procedural_matvec,
        "no backend injected, so the host-procedural-matvec flag must stay clear (#1209)"
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

// ----------------------------------------------------------------------
/// #1795 — the row-block preconditioner builder is another reduced-Schur
/// factorization entry point. It must use the same spectral PD-floor as the
/// direct dense solve, rather than a raw Cholesky, because the preconditioner
/// inverts the same collapsed decoder subspace before CG handles the explicit
/// cross-row Woodbury coupling.
#[test]
pub(crate) fn cross_row_preconditioner_build_honors_pd_floor_1795() {
    let backend = CpuBatchedBlockSolver;
    let mut sys = diagonal_arrow_fixture(2.0, 1.0);
    // With zero H_tβ blocks, the reduced Schur is exactly H_ββ. This matrix has
    // eigenvalues {+3, −1}: a bare Cholesky must reject it, while the #1038
    // spectral floor unit-deflates the collapsed direction relative to λ_max=3.
    sys.hbb = array![[1.0_f64, 2.0], [2.0, 1.0]];

    let unfloored =
        ArrowBlockDiagInverse::build(&sys, 0.0, 0.0, None, &backend, gam_gpu::GpuPolicy::Auto);
    assert!(
        matches!(unfloored, Err(ArrowSchurError::SchurFactorFailed { .. })),
        "un-floored cross-row preconditioner must surface the non-PD Schur"
    );

    let floored = ArrowBlockDiagInverse::build(
        &sys,
        0.0,
        0.0,
        Some(SPECTRAL_DEFLATION_REL_FLOOR),
        &backend,
        gam_gpu::GpuPolicy::Auto,
    )
    .expect("cross-row preconditioner must honor the spectral PD-floor");

    let rhs_t = Array1::<f64>::zeros(sys.row_offsets[sys.rows.len()]);
    let rhs_beta = array![0.25_f64, -0.5];
    let (_sol_t, sol_beta) = floored.apply(rhs_t.view(), rhs_beta.view());
    assert!(
        sol_beta.iter().all(|v| v.is_finite()),
        "floored cross-row preconditioner solve must produce finite beta components, got {sol_beta:?}"
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
/// invocations regardless of thread scheduling, the #1017 verification gate;
/// and (b) numerically equal to the sequential per-row fold up to the ULP-level
/// reordering of an otherwise-identical sum (the chunk-partial reduction
/// reassociates the same row contributions, so it agrees with the per-row
/// fold to a tight relative tolerance, not bit-for-bit). Because (b) is only
/// tolerance-equal and not bit-for-bit, the criterion ranking across candidates
/// is stable up to that reassociation margin but CAN flip a near-tie winner
/// inside it — run-to-run determinism does not by itself pin the ranking (#1211).
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

/// #1017 dense-Schur assembly parallelism: `reduce_row_schur_contributions`
/// (consumed by `build_dense_schur_direct` / `build_dense_schur_sqrt_ba`) folds
/// the per-row `-Σ_i leftᵀ·right` contributions into the `k×k` reduced Schur
/// matrix. On a CPU-only host (the `None`-tiles branch, the live path here) this
/// O(n·d·k²) reduction was the last serial step of the dense reduced-solve build;
/// at the SAE Direct-solve shape (`n` in the thousands, wide border `k`) it is
/// the dense assembly's whole cost. It now fans across rayon over fixed CHUNK=64
/// row chunks (each chunk reduces in row order into a private partial; partials
/// folded into `schur` in chunk order).
///
/// Assert (a) DETERMINISM — two independent parallel builds are bit-for-bit
/// identical regardless of thread scheduling (the #1017 verification gate); and
/// (b) EQUIVALENCE with the in-place serial per-row reduction up to ULP-scale
/// chunk-boundary reassociation of an otherwise-identical sum (the same bar the
/// streaming `accumulate_chunk` and per-row matvec parity tests hold). Note (a)
/// only fixes the result run-to-run; because (b) is tolerance-equal not
/// bit-for-bit with serial, the criterion ranking is stable up to the
/// reassociation margin and a near-tie winner inside it can flip (#1211).
#[test]
pub(crate) fn parallel_dense_schur_reduction_deterministic_and_matches_sequential() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64; // > MIN → trips the parallel CPU fold
    let d = 5usize;
    let k = 48usize;
    let sys = dense_direct_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1e-6;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");

    for kind in [SchurReductionKind::Direct, SchurReductionKind::SqrtBa] {
        // Seed `H_ββ + ridge·I` exactly as the dense builders do.
        let seed = || {
            let mut s = sys.effective_penalty_op().to_dense();
            for j in 0..k {
                s[[j, j]] += ridge_beta;
            }
            s
        };

        // (a) Determinism: two independent parallel reductions are bit-identical.
        let mut s_a = seed();
        reduce_row_schur_contributions(
            &sys,
            &htt_factors,
            &backend,
            kind,
            &mut s_a,
            gam_gpu::GpuPolicy::Auto,
        )
        .expect("parallel reduction a");
        let mut s_b = seed();
        reduce_row_schur_contributions(
            &sys,
            &htt_factors,
            &backend,
            kind,
            &mut s_b,
            gam_gpu::GpuPolicy::Auto,
        )
        .expect("parallel reduction b");
        for a in 0..k {
            for b in 0..k {
                assert_eq!(
                    s_a[[a, b]].to_bits(),
                    s_b[[a, b]].to_bits(),
                    "{kind:?}: parallel dense-Schur reduction must be deterministic \
                     run-to-run at ({a},{b})"
                );
            }
        }

        // (b) Equivalence with the in-place serial per-row reduction.
        let mut s_ser = seed();
        for (i, row) in sys.rows.iter().enumerate() {
            subtract_row_schur_contribution(
                &sys,
                i,
                row,
                htt_factors.factor(i),
                &backend,
                kind,
                &mut s_ser,
            )
            .expect("serial per-row reduction");
        }
        let scale = s_ser.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
        let mut max_rel = 0.0_f64;
        for a in 0..k {
            for b in 0..k {
                max_rel = max_rel.max((s_a[[a, b]] - s_ser[[a, b]]).abs() / scale);
            }
        }
        // The parallel reduction folds per-thread partials in a different order
        // than the serial per-row reduction; f64 non-associativity means the gap
        // scales with the worker/chunk count, so a 1e-15 bound that held on a
        // low-core box is exceeded (~1e-14) on a 64+-core A100 node. Tolerate the
        // unavoidable reassociation at a still-tight 1e-12 — far below any real
        // divergence — rather than pin a core-count-dependent bit pattern.
        assert!(
            max_rel < 1e-12,
            "{kind:?}: parallel vs serial dense-Schur reduction must agree to \
             reassociation error (rel {max_rel:e})"
        );
    }
}

/// #1017 cluster-Jacobi build parallelism: the per-cluster `b×b` Schur block
/// assembly in `ClusterJacobiPreconditioner::build_from_column_groups` runs the
/// independent rows over fixed 64-row chunks above `SCHUR_MATVEC_PARALLEL_ROW_MIN`
/// and folds chunk partials in chunk order, exactly like `build_block_jacobi`.
/// This pins the parallel-fold preconditioner against (a) bit-identical
/// run-to-run determinism and (b) an independent serial row-order reference of
/// the same Schur block (tolerance-equal, not bit-for-bit). (a) makes the
/// preconditioner invariant to the thread SCHEDULE run-to-run; it does not make
/// it bit-identical to serial, so a criterion ranking the preconditioner feeds
/// is stable only up to the reassociation margin — a near-tie can still flip
/// (#1211).
#[test]
pub(crate) fn cluster_jacobi_build_deterministic_and_matches_serial() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64; // > MIN → trips the parallel CPU fold
    let d = 5usize;
    let k = 48usize; // single cluster, b = k ≤ CLUSTER_JACOBI_MAX_CLUSTER → Chol path
    let sys = dense_direct_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1e-6;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let cols: Vec<usize> = (0..k).collect();
    let col_groups = vec![cols.clone()];

    // A deterministic probe vector to drive `apply` through the assembled factor.
    let r: Array1<f64> =
        Array1::from_iter((0..k).map(|j| 0.1 * ((j + 1) as f64).sin() - 0.03 * j as f64));

    // (a) Determinism: two independent parallel builds apply bit-identically.
    let p_a = ClusterJacobiPreconditioner::build_from_column_groups(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        &col_groups,
    )
    .expect("cluster build a");
    let p_b = ClusterJacobiPreconditioner::build_from_column_groups(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        &col_groups,
    )
    .expect("cluster build b");
    let out_a = p_a.apply(&r);
    let out_b = p_b.apply(&r);
    for j in 0..k {
        assert_eq!(
            out_a[j].to_bits(),
            out_b[j].to_bits(),
            "cluster-Jacobi build must be deterministic run-to-run at {j}"
        );
    }

    // (b) Serial reference: assemble the same `b×b` cluster Schur block in
    // strict row order, factor with the same faer LLT, and solve `r` through it.
    let b = k;
    let mut s_ref = Array2::<f64>::zeros((b, b));
    sys.penalty_subblock_add(&cols, &mut s_ref);
    for bi in 0..b {
        s_ref[[bi, bi]] += ridge_beta;
    }
    let mut col_vec = Array1::<f64>::zeros(d);
    let mut solved_cols = Array2::<f64>::zeros((d, b));
    for (row_idx, row) in sys.rows.iter().enumerate() {
        for bj in 0..b {
            let gj = cols[bj];
            for c in 0..d {
                col_vec[c] = row.htbeta[[c, gj]];
            }
            let solved = backend.solve_block_vector(htt_factors.factor(row_idx), col_vec.view());
            for c in 0..d {
                solved_cols[[c, bj]] = solved[c];
            }
        }
        for bi in 0..b {
            let gi = cols[bi];
            for bj in 0..b {
                let mut acc = 0.0;
                for c in 0..d {
                    acc += row.htbeta[[c, gi]] * solved_cols[[c, bj]];
                }
                s_ref[[bi, bj]] -= acc;
            }
        }
    }
    // Mirror the build's symmetrize + faer LLT solve of the probe.
    for i in 0..b {
        for j in 0..i {
            let v = 0.5 * (s_ref[[i, j]] + s_ref[[j, i]]);
            s_ref[[i, j]] = v;
            s_ref[[j, i]] = v;
        }
    }
    let llt = {
        use faer::Side;
        let view = FaerArrayView::new(&s_ref);
        FaerLlt::new(view.as_ref(), Side::Lower).expect("reference Schur block must be PD")
    };
    let solved_ref = {
        use faer::linalg::solvers::Solve;
        let mut rhs = r.clone();
        let stride = rhs.strides()[0];
        let len = rhs.len();
        // SAFETY: `rhs` is a contiguous owned `Array1<f64>` of `len` elements that
        // outlives this borrow; `as_mut_ptr()` is valid and aligned for `len`
        // reads. We view it as a `len × 1` column-major matrix whose row stride is
        // the array's element stride; with a single column the column stride is
        // never dereferenced, so `0` is sound. `rhs` is not aliased while the view
        // is live (it is only read through `llt.solve`).
        let rhs_mat = unsafe { faer::MatRef::from_raw_parts(rhs.as_mut_ptr(), len, 1, stride, 0) };
        let s = llt.solve(rhs_mat);
        Array1::from_iter((0..b).map(|i| s[(i, 0)]))
    };
    let scale = solved_ref
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    let mut max_rel = 0.0_f64;
    for j in 0..k {
        max_rel = max_rel.max((out_a[j] - solved_ref[j]).abs() / scale);
    }
    assert!(
        max_rel < 1e-12,
        "parallel cluster-Jacobi apply must match the serial row-order reference \
         to reassociation error (rel {max_rel:e})"
    );
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
/// invocations regardless of thread scheduling (the #1017 gate); and (b) equal
/// to the sequential row-order fold — bit-identical on the disjoint `y_t` writes
/// and within ULP-scale reassociation on the cross-row `y_beta` sum. Since the
/// `y_beta` sum is only tolerance-equal to serial (not bit-for-bit), the
/// criterion ranking is stable up to that margin but a near-tie winner inside it
/// can flip; run-to-run determinism alone does not pin the ranking (#1211).
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
    let precond = ArrowBlockDiagInverse::build(
        &sys,
        ridge_t,
        ridge_beta,
        None,
        &backend,
        gam_gpu::GpuPolicy::Auto,
    )
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

    // Serial reference: a hand GEMV `hbb·x + ridge·x`, independent of
    // `penalty_matvec_add` (which itself now parallelizes at this border width,
    // so it can no longer serve as the serial oracle for either stage).
    let mut serial = vec![0.0_f64; k];
    for a in 0..k {
        let mut acc = 0.0_f64;
        for b in 0..k {
            acc += sys.hbb[[a, b]] * xs[b];
        }
        serial[a] = acc + ridge * xs[a];
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

/// `penalty_matvec_add` is the serial `H_ββ·x` accumulate left inside the
/// per-CG-iteration cross-row matvec (`arrow_cross_row_matvec`); at the wide SAE
/// border it fans over output rows. Because each `y[a] += Σ_b hbb[a,b]·x[b]` is
/// one thread's own dot in the same `b` order as serial, the parallel accumulate
/// is **bit-identical** to serial (not merely deterministic), so the criterion
/// ranking cannot move. Also pins the accumulate semantics — the parallel path
/// must ADD into a pre-seeded `y`, not overwrite it.
#[test]
pub(crate) fn parallel_penalty_matvec_add_bit_identical_to_serial() {
    let k = 576usize; // ≥ SCHUR_PROLOGUE_PARALLEL_K_MIN: trips the parallel GEMV
    assert!(k >= SCHUR_PROLOGUE_PARALLEL_K_MIN);
    let d = 4usize;
    let n = 8usize; // rows below the per-row floor: isolate the GEMV branch
    let sys = dense_arrow_system(n, d, k);
    let x = Array1::from_iter((0..k).map(|a| 0.6 * (a as f64 * 0.23).sin() + 0.11));
    let xs = x.as_slice().unwrap();
    // Pre-seed `y` so the accumulate (not overwrite) semantics are exercised.
    let seed: Vec<f64> = (0..k).map(|a| (a as f64 * 0.017).cos() - 0.3).collect();

    // Hand serial reference: `y = seed + hbb·x`.
    let mut serial = seed.clone();
    for a in 0..k {
        let mut acc = 0.0_f64;
        for b in 0..k {
            acc += sys.hbb[[a, b]] * xs[b];
        }
        serial[a] += acc;
    }

    // Two parallel calls: bit-identical to serial AND to each other.
    let mut par_a = seed.clone();
    sys.penalty_matvec_add(xs, &mut par_a);
    let mut par_b = seed.clone();
    sys.penalty_matvec_add(xs, &mut par_b);
    for a in 0..k {
        assert_eq!(
            par_a[a].to_bits(),
            serial[a].to_bits(),
            "parallel penalty_matvec_add must be bit-identical to serial at index {a}"
        );
        assert_eq!(
            par_a[a].to_bits(),
            par_b[a].to_bits(),
            "parallel penalty_matvec_add must be run-to-run deterministic at index {a}"
        );
    }
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
    // is the exact adjoint. Mirrors src/terms/sae/manifold/mod.rs:6028.
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
        a_phi: std::sync::Arc::from(a_phi.clone().into_boxed_slice()),
        local_jac: std::sync::Arc::from(local_jac.clone().into_boxed_slice()),
        smooth_blocks: Vec::new(),
        sparse_g_blocks: Vec::new(),
        frame: None,
    });
    (sys, a_phi, local_jac)
}

/// Build a WELL-POSED device-equipped SAE system for the end-to-end engagement
/// parity tests: PD reduced Schur, matching device/dense `H_ββ`, materialized
/// cross-block `H_tβ`, and deterministic nonzero gradients. Shared by the Direct
/// and InexactPCG engagement tests so both exercise the identical operator.
///
/// Shape `k = n_atoms·p = 64 ≥ DEVICE_LOOP_MIN_P (32)`; the work predicate
/// `n·(2·d·k + d²)·cg_iters` clears `MATVEC_OFFLOAD_FLOPS_MIN` by orders of
/// magnitude. Modest `n` keeps the CPU reference + dense parity check cheap.
pub(crate) fn well_posed_device_sae_system_1551() -> (ArrowSchurSystem, usize, usize) {
    let n = 512usize;
    let q = 4usize; // per-row latent depth d
    let p = 8usize;
    let n_atoms = 8usize;
    let m_active = 4usize;
    let (mut sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);

    // The device non-framed `H_ββ` is assembled from `data.sparse_g_blocks` (as
    // `G ⊗ I_p`), NOT from `sys.hbb` (which only the dense reference reads). For a
    // sound parity fixture the two MUST encode the same matrix. In
    // `sae_structured_system` each atom owns `p` consecutive β slots (μ-space
    // index = atom, so `m_i = 1`), so a dominant diagonal `H_ββ` is one 1×1
    // `SparseGBlock` per atom. Make it strongly diagonally dominant so the reduced
    // Schur `S = (H_ββ + ρI) − Σ_i H_βt^(i)(H_tt^(i)+ρI)⁻¹ H_tβ^(i)` stays PD once
    // the n=512-row subtraction accumulates (otherwise the device correctly fails
    // loud on a non-positive Schur Jacobi diagonal — that fail-loud IS the #1551
    // contract, just not what this parity fixture is exercising).
    let hbb_diag = (n as f64) + 1000.0;
    let mut sparse_g_blocks = Vec::with_capacity(n_atoms);
    let mut new_hbb = Array2::<f64>::zeros((sys.k, sys.k));
    for atom in 0..n_atoms {
        sparse_g_blocks.push(SparseGBlock {
            row_off: atom,
            col_off: atom,
            data: ndarray::array![[hbb_diag]],
        });
        // `G ⊗ I_p`: the 1×1 μ-block at (atom, atom) puts `hbb_diag` on the
        // diagonal of all p channels of this atom's β slots.
        for c in 0..p {
            new_hbb[[atom * p + c, atom * p + c]] = hbb_diag;
        }
    }
    sys.hbb = new_hbb;
    // Reinstall the device payload carrying the matching sparse-G data Hessian.
    {
        let device = sys
            .device_sae_pcg
            .as_ref()
            .expect("fixture installs device data");
        let new_device = DeviceSaePcgData {
            p: device.p,
            beta_dim: device.beta_dim,
            a_phi: std::sync::Arc::clone(&device.a_phi),
            local_jac: std::sync::Arc::clone(&device.local_jac),
            smooth_blocks: device.smooth_blocks.clone(),
            sparse_g_blocks,
            frame: None,
        };
        sys.set_device_sae_pcg_data(new_device);
    }

    // PARITY-ORACLE CONSISTENCY: `solve_arrow_newton_step_dense_reference` reads
    // the cross-block `H_tβ` from `row.htbeta` DIRECTLY — it does not invoke the
    // installed `htbeta_matvec` operator. But `sae_structured_system` ships the
    // coupling ONLY as that matrix-free operator (`row.htbeta` is all-zeros), so
    // an unmaterialized fixture makes the dense reference solve a DECOUPLED system
    // (H_tβ ≡ 0) while the device solves the true coupled one — the parity gap is
    // then the omitted coupling term, not a kernel/conditioning artifact. Materialize
    // the operator into each `row.htbeta` (exact for a linear operator: probe with
    // unit columns). With `htbeta_dense_supplement == false` the production apply
    // (`sys_htbeta_apply_row`) still uses the operator ONLY, so the device/CPU
    // matrix-free path is unchanged; the dense reference now reads the identical
    // operator. All three paths (device PCG, CPU reduced, dense reference) then
    // encode one and the same `H_tβ`.
    assert!(
        !sys.htbeta_dense_supplement,
        "fixture must keep dense-supplement OFF so the matrix-free apply uses the \
         operator only (row.htbeta is the dense ECHO for the reference, not a second \
         additive slab)"
    );
    let materialized: Vec<Array2<f64>> = (0..sys.rows.len())
        .map(|i| {
            sys_htbeta_materialize_row(&sys, i, &sys.rows[i])
                .expect("materialize row H_tβ from installed operator")
        })
        .collect();
    for (i, mat) in materialized.into_iter().enumerate() {
        sys.rows[i].htbeta = mat;
    }

    // The fixture ships zero gradients (trivial zero step); install deterministic
    // nonzero g_t / g_β so the solved Δ is a real, discriminating vector.
    for (i, row) in sys.rows.iter_mut().enumerate() {
        for r in 0..q {
            row.gt[r] = 0.1 * (((i + 1) * (r + 2)) as f64 * 0.013).sin();
        }
    }
    for a in 0..sys.k {
        sys.gb[a] = 0.05 * ((a as f64 + 1.0) * 0.021).cos() - 0.02;
    }
    (sys, n, q)
}

/// #2660 ALGORITHM SELECTION — a Direct SAE solve must have one canonical dense
/// owner even when the system carries device-PCG data and clears that lane's
/// economic gate. Automatic Direct is bounded by `DIRECT_SOLVE_MAX_K`, so it
/// already needs the exact dense Schur factor for evidence; running matrix-free
/// PCG first would duplicate the dominant reduction and, before #2660, solve an
/// unquotiented/unfloored operator whose result disagreed with that factor.
///
/// Drive the public production core with the exact formerly-eligible fixture.
/// The profitable stacked dense-Schur sequence may still run on the device, so
/// generic device-ownership telemetry and generic PCG counters cannot distinguish
/// it from matrix-free PCG (dense Steihaug may legitimately use the latter). Pin
/// the algorithm selector itself instead: Direct must neither prepare a resident
/// SAE-PCG frame nor select matrix-free PCG. Tolerance-based parity against the
/// dense reference then proves the selected dense owner remains numerically
/// exact without requiring bit identity across independent Auto-dispatched
/// assembly executions.
#[test]
pub(crate) fn sae_direct_uses_canonical_dense_owner_not_matrix_free_pcg_2660() {
    let (sys, n, q) = well_posed_device_sae_system_1551();

    let policy = gam_gpu::policy::GpuDispatchPolicy::default();
    assert!(
        policy.reduced_schur_matvec_should_offload(n, sys.k, q, DEFAULT_PCG_MAX_ITERATIONS),
        "fixture must clear the reduced-Schur offload gate so the device path is eligible"
    );

    let options = ArrowSolveOptions::direct();
    let ridge_t = 1e-7;
    let ridge_beta = 1e-6;

    let resident = prepare_sae_resident_frame(&sys, &options, None)
        .expect("Direct algorithm-selection probe");
    assert!(
        resident.is_none(),
        "#2660: Direct prepared the forbidden resident matrix-free SAE-PCG owner"
    );

    let (_, _, diagnostics) =
        solve_arrow_newton_step_core(&sys, ridge_t, ridge_beta, &options)
            .expect("SAE Direct production-core solve");
    assert!(
        !diagnostics.selected_matrix_free_pcg,
        "#2660: Direct selected the forbidden matrix-free PCG algorithm"
    );

    let artifacts = solve_arrow_newton_step_artifacts(&sys, ridge_t, ridge_beta, &options)
        .expect("SAE Direct canonical artifacts solve");
    assert!(
        artifacts.schur_factor.is_some(),
        "#2660: Direct must retain the same canonical dense factor for evidence"
    );
    // Parity (holds on every host): the produced Newton step must match the dense
    // joint-system reference. On a GPU host this is the device==CPU parity gate;
    // on a CPU host it pins the matrix-free reduced solve to the dense oracle.
    let reference = crate::gpu_kernels::arrow_schur::solve_arrow_newton_step_dense_reference(
        &sys, ridge_t, ridge_beta,
    )
    .expect("dense reference solve");
    let db_scale = reference
        .delta_beta
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    let mut max_db_rel = 0.0_f64;
    for a in 0..sys.k {
        max_db_rel =
            max_db_rel.max((artifacts.delta_beta[a] - reference.delta_beta[a]).abs() / db_scale);
    }
    assert!(
        max_db_rel <= 1e-7,
        "#2660 SAE Direct canonical Δβ parity vs dense reference: \
         max_rel={max_db_rel:e} (>1e-7)"
    );
    let dt_scale = reference
        .delta_t
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    let mut max_dt_rel = 0.0_f64;
    for i in 0..artifacts.delta_t.len() {
        max_dt_rel = max_dt_rel.max((artifacts.delta_t[i] - reference.delta_t[i]).abs() / dt_scale);
    }
    assert!(
        max_dt_rel <= 1e-7,
        "#2660 SAE Direct canonical Δt parity vs dense reference: \
         max_rel={max_dt_rel:e} (>1e-7)"
    );
}

/// #1551/#1209 PRODUCTION ENGAGEMENT (InexactPCG mode) — the LARGE-K regime
/// (`K > DIRECT_SOLVE_MAX_K`) that `ArrowSolverMode::automatic` routes to
/// `InexactPCG`, which is where the device matters MOST. The InexactPCG branch
/// of `solve_arrow_newton_step_core` runs the device matrix-free SAE PCG when the
/// trust radius is unbounded (the SAE inner-solve default). This pins TWO
/// contracts the Direct test cannot:
///   1. ENGAGEMENT (`used_device_arrow == true` on a CUDA host) for the InexactPCG
///      seam specifically — a separate code path from the Direct seam.
///   2. FAIL-LOUD routing (#1209): the branch must NOT swallow a device kernel
///      fault and silently continue on the CPU with `used_device_arrow == false`.
///      A genuine `Unavailable` decline still falls through transparently.
/// Parity vs the dense reference holds on every host (CPU oracle == device).
#[test]
pub(crate) fn sae_inexact_pcg_inner_solve_engages_device_and_matches_cpu_1551() {
    let (sys, n, q) = well_posed_device_sae_system_1551();

    let policy = gam_gpu::policy::GpuDispatchPolicy::default();
    assert!(
        policy.reduced_schur_matvec_should_offload(n, sys.k, q, DEFAULT_PCG_MAX_ITERATIONS),
        "fixture must clear the reduced-Schur offload gate so the device path is eligible"
    );

    // `inexact_pcg()` defaults to an unbounded trust radius (f64::INFINITY), so the
    // device matrix-free SAE PCG branch of `solve_arrow_newton_step_core` is the
    // one exercised here (NOT the Direct seam).
    let options = ArrowSolveOptions::inexact_pcg();
    assert_eq!(
        options.trust_region.radius,
        f64::INFINITY,
        "InexactPCG default must keep the unbounded trust radius that authorizes the \
         device matrix-free SAE PCG branch"
    );
    let ridge_t = 1e-7;
    let ridge_beta = 1e-6;

    let artifacts = solve_arrow_newton_step_artifacts(&sys, ridge_t, ridge_beta, &options)
        .expect("SAE InexactPCG artifacts solve");
    assert!(
        artifacts.pcg_diagnostics.selected_matrix_free_pcg,
        "#2660: InexactPCG did not select its matrix-free reduced-Schur owner"
    );

    if !gpu_available_or_fail() {
        assert!(
            !artifacts.pcg_diagnostics.used_device_arrow,
            "no CUDA device present, yet the InexactPCG step was flagged device-served"
        );
    } else {
        // CUDA present + the fixture clears the gate ⇒ the InexactPCG inner solve
        // MUST run on the device. After the #1209/#1551 fail-loud routing fix, a
        // device kernel fault would surface as an Err (caught by `.expect` above);
        // a genuine decline would fall through to CPU. Either way a silent
        // device→CPU fallback under a healthy device is the failure we forbid.
        assert!(
            artifacts.pcg_diagnostics.used_device_arrow,
            "#1551: CUDA device present and the offload gate cleared, but the SAE \
             InexactPCG inner solve did NOT engage the device \
             (used_device_arrow=false) — the device path silently fell back to CPU"
        );
    }

    // Parity (holds on every host): the produced Newton step must match the dense
    // joint-system reference, exactly as in the Direct test (same well-posed system).
    let reference = crate::gpu_kernels::arrow_schur::solve_arrow_newton_step_dense_reference(
        &sys, ridge_t, ridge_beta,
    )
    .expect("dense reference solve");
    let db_scale = reference
        .delta_beta
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    let mut max_db_rel = 0.0_f64;
    for a in 0..sys.k {
        max_db_rel =
            max_db_rel.max((artifacts.delta_beta[a] - reference.delta_beta[a]).abs() / db_scale);
    }
    assert!(
        max_db_rel <= 1e-7,
        "#1551 SAE InexactPCG Δβ parity vs dense reference: max_rel={max_db_rel:e} (>1e-7) \
         (device-served={})",
        artifacts.pcg_diagnostics.used_device_arrow
    );
    let dt_scale = reference
        .delta_t
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    let mut max_dt_rel = 0.0_f64;
    for i in 0..artifacts.delta_t.len() {
        max_dt_rel = max_dt_rel.max((artifacts.delta_t[i] - reference.delta_t[i]).abs() / dt_scale);
    }
    assert!(
        max_dt_rel <= 1e-7,
        "#1551 SAE InexactPCG Δt parity vs dense reference: max_rel={max_dt_rel:e} (>1e-7)"
    );
}

/// #1209 MUTUAL EXCLUSION — `used_device_arrow` (the matvec/solve genuinely ran
/// on the device) and `injected_host_procedural_matvec` (a host Rust/Rayon
/// reduced-Schur matvec closure was injected and ran on the CPU) describe
/// MUTUALLY EXCLUSIVE execution facts: a single solve cannot have run its matvec
/// both on the device and on the host. The production core entry
/// `solve_arrow_newton_step_core` injects a host procedural matvec via
/// `maybe_inject_gpu_schur_matvec` for an admitted InexactPCG SAE system — but
/// the re-entered solve may itself take the genuinely device-resident SAE PCG
/// branch (`device_sae_pcg` present) and never consume that injected closure. A
/// naive unconditional stamp would then report BOTH flags (a contradiction, and
/// an `injected_host_procedural_matvec` that is simply wrong — the matvec ran on
/// the device). This pins that the two flags are never simultaneously set,
/// driven through the public production path on a CUDA host.
#[test]
pub(crate) fn device_arrow_and_host_procedural_matvec_flags_are_mutually_exclusive_1209() {
    let (sys, _n, _q) = well_posed_device_sae_system_1551();
    let ridge_t = 1e-7;
    let ridge_beta = 1e-6;

    // Exercise the explicit InexactPCG entry (the one that injects a host
    // procedural matvec via `maybe_inject_gpu_schur_matvec`) and the Direct entry
    // through the public core.
    let on_cuda = gpu_available_or_fail();
    let mut inexact_used_device = false;
    for options in [
        ArrowSolveOptions::inexact_pcg(),
        ArrowSolveOptions::direct(),
    ] {
        let mode = options.mode;
        let (_dt, _db, diag) =
            solve_arrow_newton_step_core(&sys, ridge_t, ridge_beta, &options).expect("core solve");
        assert!(
            !(diag.used_device_arrow && diag.injected_host_procedural_matvec),
            "#1209: used_device_arrow and injected_host_procedural_matvec are mutually \
             exclusive execution facts but BOTH were set (mode={mode:?}) — a single solve \
             cannot run its reduced-Schur matvec on the device AND as a host procedural \
             closure"
        );
        if mode == ArrowSolverMode::InexactPCG && diag.used_device_arrow {
            inexact_used_device = true;
        }
    }

    // NON-VACUITY: on a CUDA host the InexactPCG path is EXACTLY the regression
    // scenario — `maybe_inject_gpu_schur_matvec` injects a host matvec AND the
    // re-entered solve takes the device-resident SAE PCG branch. If the device
    // branch never engaged the mutual-exclusion assertion would pass trivially,
    // so confirm the contradictory pre-condition (device-served InexactPCG) was
    // actually reached. (The injection itself fires whenever the offload gate
    // clears, which the well-posed fixture is built to do.)
    if on_cuda {
        assert!(
            inexact_used_device,
            "#1209: fixture must reach the device-served InexactPCG path on a CUDA host \
             so the mutual-exclusion check is non-vacuous"
        );
    }
}

/// The CPU-resident SAE reduced-Schur matvec (#1017) must compute the SAME
/// `S·x` as the generic per-row `apply → solve → transpose` path, up to f64
/// reassociation. This is the residency correctness gate: a resident matvec
/// that changed the reduced operator (beyond f64 reassociation) would change the
/// Newton step and could move the criterion ranking — a correctness regression,
/// not a speedup. (The allowed f64 reassociation can itself still flip a
/// near-tie ranking within the margin; this gate bounds the operator to that
/// margin, it does not promise an exact no-move — see #1211.)
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

/// #1033 large-n sharing invariant (solver side). The per-row support `a_phi`
/// and local Jacobians `local_jac` are held as `Arc<[…]>` so the assembler can
/// hand consumers the SAME backing allocation instead of a second full
/// `O(n·q·p)` clone. The host-operator (`gam_sae::SaeKroneckerRows`) ↔ solver
/// (`DeviceSaePcgData`) cross-crate sharing half of this contract lives in
/// `gam-sae` (`device_and_kron_rows_share_backing_alloc_1033`) — only that crate
/// can see both types, since `gam-solve` cannot depend on `gam-sae` (the edge
/// runs the other way after the #1521 carve). This half pins the solver-internal
/// half: `DeviceSaePcgData::a_phi_shared()` must hand back a refcount bump of the
/// data's own `a_phi` (`Arc::ptr_eq`), not a fresh deep clone. A regression that
/// reverts to a `Vec` deep-clone would double the always-resident per-row
/// footprint at the LLM shape (p≈5120) and fail `Arc::ptr_eq` here.
#[test]
pub(crate) fn device_a_phi_shared_is_refcount_bump_not_clone_1033() {
    let p = 6usize;
    let a_phi: std::sync::Arc<[Vec<(usize, f64)>]> = std::sync::Arc::from(
        vec![vec![(0usize, 2.0f64), (12, 1.0)], vec![(0, 0.5)]].into_boxed_slice(),
    );
    let jac: std::sync::Arc<[Vec<f64>]> =
        std::sync::Arc::from(vec![vec![1.0; 4 * p], vec![2.0; 4 * p]].into_boxed_slice());
    let device = DeviceSaePcgData {
        p,
        beta_dim: 6,
        a_phi: std::sync::Arc::clone(&a_phi),
        local_jac: std::sync::Arc::clone(&jac),
        smooth_blocks: Vec::new(),
        sparse_g_blocks: Vec::new(),
        frame: None,
    };
    // a_phi_shared is O(1) — same backing buffer, not a deep clone.
    let reshare = device.a_phi_shared();
    assert!(
        std::sync::Arc::ptr_eq(&reshare, &device.a_phi),
        "a_phi_shared must hand back the SAME allocation, not a re-clone"
    );
    assert!(
        std::sync::Arc::ptr_eq(&reshare, &a_phi),
        "a_phi_shared must alias the assembler's original a_phi allocation"
    );
}

/// #1017/#2230 residency measurement: `operand_byte_report` must categorise the
/// per-solve host→device operand bytes correctly on BOTH matrix-free sub-lanes,
/// so the a100 job's log numbers are trustworthy. Legacy (`frame = None`) carries
/// the sparse `a_phi`/`local_jac` and zero `row_htbeta`; framed carries a dense
/// per-row `row_htbeta` (the 34MiB-vs-31GiB discriminator the #2230 report flags).
#[test]
pub(crate) fn sae_pcg_operand_byte_report_categorises_both_lanes_1017() {
    let p = 5usize;
    // Legacy sparse lane: 2 rows, supports of 3 and 2 atoms; local_jac 4+6 f64.
    let a_phi: std::sync::Arc<[Vec<(usize, f64)>]> = std::sync::Arc::from(
        vec![
            vec![(0usize, 1.0), (2, 0.5), (7, -0.3)],
            vec![(1usize, 1.0), (4, 0.2)],
        ]
        .into_boxed_slice(),
    );
    let jac: std::sync::Arc<[Vec<f64>]> =
        std::sync::Arc::from(vec![vec![1.0; 4], vec![2.0; 6]].into_boxed_slice());
    let legacy = DeviceSaePcgData {
        p,
        beta_dim: 12,
        a_phi: std::sync::Arc::clone(&a_phi),
        local_jac: std::sync::Arc::clone(&jac),
        smooth_blocks: Vec::new(),
        sparse_g_blocks: Vec::new(),
        frame: None,
    };
    let r = legacy.operand_byte_report();
    assert!(!r.framed, "frame = None must report the legacy sparse lane");
    assert_eq!(r.a_phi_pairs, 5, "3 + 2 support pairs");
    assert_eq!(r.a_phi_bytes, 5 * std::mem::size_of::<(usize, f64)>());
    assert_eq!(r.local_jac_elems, 10);
    assert_eq!(r.local_jac_bytes, 10 * 8);
    assert_eq!(
        r.row_htbeta_bytes, 0,
        "legacy lane has no dense per-row cross"
    );
    assert_eq!(r.frame_blocks_bytes, 0);
    assert_eq!(r.total_bytes, r.a_phi_bytes + r.local_jac_bytes);

    // Framed dense lane: 3 rows, two carrying a length-4 dense cross, one empty.
    let frame = DeviceSaeFrameData {
        ranks: vec![2, 2],
        basis_sizes: vec![3, 3],
        border_offsets: vec![0, 6],
        frame_blocks: Vec::new(),
        smooth_ranks: Vec::new(),
        row_htbeta: vec![vec![0.0; 4], vec![0.0; 4], Vec::new()],
    };
    let framed = DeviceSaePcgData {
        p,
        beta_dim: 12,
        a_phi: std::sync::Arc::clone(&a_phi),
        local_jac: std::sync::Arc::clone(&jac),
        smooth_blocks: Vec::new(),
        sparse_g_blocks: Vec::new(),
        frame: Some(frame),
    };
    let rf = framed.operand_byte_report();
    assert!(rf.framed, "frame = Some must report the framed dense lane");
    assert_eq!(
        rf.row_htbeta_rows, 2,
        "two rows carry a non-empty cross slab"
    );
    assert_eq!(rf.row_htbeta_bytes, 8 * 8, "4 + 4 f64 across the two rows");
    assert_eq!(
        rf.total_bytes,
        rf.a_phi_bytes + rf.local_jac_bytes + rf.row_htbeta_bytes,
        "total must fold the framed dense cross into the per-solve upload"
    );
}

/// #1033 frames-engaged assembly guard: `set_device_sae_pcg_data` must NOT panic
/// when the frames-engaged builder (`build_framed_device_sae_data`) hands it a
/// `DeviceSaePcgData` whose full-`B` per-row `a_phi`/`local_jac` slabs are left
/// intentionally EMPTY (the per-row cross block rides `frame.frame_blocks`
/// instead). Before the fix the install unconditionally asserted
/// `a_phi.len() == rows.len()` and `local_jac.len() == rows.len()`, so EVERY
/// frames-engaged SAE assembly (decoder rank < p — the common large-output case)
/// panicked at install; it was dormant only because no test exercised a
/// frame-activating shape, and it surfaced while profiling a real OLMo l18 fit.
///
/// This pins both halves of the fix: (1) the relaxed length contract — the
/// per-row-slab asserts apply ONLY when `frame.is_none()`, so a framed payload
/// installs without panicking; and (2) the consumer contract — with the slabs
/// empty the CPU-resident reduced-Schur factor must DECLINE to build
/// (`SaeResidentReducedSchur::build → None`), so the solve falls back to the
/// generic per-row matvec rather than relocating the panic to an empty-slab
/// index. Reverting the assert gate makes `set_device_sae_pcg_data` panic here.
#[test]
pub(crate) fn framed_device_sae_pcg_install_tolerates_empty_per_row_slabs_1033() {
    let n = 4usize;
    let q = 3usize;
    let p = 5usize;
    let n_atoms = 2usize;
    let k = n_atoms * p;
    let mut sys = ArrowSchurSystem::new(n, q, k);
    // SPD per-row H_tt so the resident factor COULD build if the slabs were
    // populated — isolating the empty-slab decline as the property under test
    // (a degenerate H_tt would let the factor fail for an unrelated reason).
    for i in 0..n {
        let mut htt = Array2::<f64>::zeros((q, q));
        for r in 0..q {
            htt[[r, r]] = (q as f64) + 2.0 + i as f64;
        }
        sys.rows[i].htt = htt;
        sys.rows[i].gt = Array1::<f64>::zeros(q);
    }

    // A frames-engaged device payload: the per-row cross block rides
    // `frame.frame_blocks`/`frame.row_htbeta`, so the full-`B` `a_phi`/`local_jac`
    // slabs are EMPTY — exactly what `build_framed_device_sae_data` produces.
    let frame = DeviceSaeFrameData {
        ranks: vec![p; n_atoms],
        basis_sizes: vec![1; n_atoms],
        border_offsets: vec![0, p], // prefix sum of M_k·r_k = 1·p per atom
        frame_blocks: Vec::new(),
        smooth_ranks: Vec::new(),
        row_htbeta: vec![Vec::new(); n],
    };
    let device = DeviceSaePcgData {
        p,
        beta_dim: k,
        a_phi: std::sync::Arc::from(Vec::<Vec<(usize, f64)>>::new().into_boxed_slice()),
        local_jac: std::sync::Arc::from(Vec::<Vec<f64>>::new().into_boxed_slice()),
        smooth_blocks: Vec::new(),
        sparse_g_blocks: Vec::new(),
        frame: Some(frame),
    };

    // The core of #1033: this install must NOT panic on the empty slabs.
    sys.set_device_sae_pcg_data(device);

    // Consumer contract: the empty-slab framed payload must make the CPU-resident
    // reduced-Schur factor decline (None) → generic per-row matvec fallback, so
    // no consumer ever indexes the empty `a_phi`/`local_jac`.
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, q, false)
        .expect("SPD per-row blocks must factor");
    let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend);
    assert!(
        resident.is_none(),
        "frames-engaged empty-slab payload must decline the resident factor and \
         fall back to the generic matvec, not index the empty per-row slabs"
    );
}

/// The #1017 SAE-resident scalar Jacobi (built from the staged `(L_i, Y_i)`
/// factors in one support-sparse pass) must produce the SAME reduced-Schur
/// diagonal — hence the SAME `BlockFactor::Scalar` inverses — as the generic
/// per-column probe-and-solve `build_scalar_jacobi`. A diverging
/// preconditioner (beyond f64 reassociation) would change the PCG iterate and
/// could move the criterion ranking. (Even the matching preconditioner is only
/// tolerance-equal to the generic build, so a near-tie ranking can still flip
/// within that margin — this is not an exact no-move guarantee, see #1211.)
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

/// #1017 SAE-resident scalar-Jacobi col-dot hoist: the per-channel column dot
/// `Σ_r L_i[r·p+j]·Y_i[r·p+j]` depends only on the row, not the support entry,
/// so the builder now computes it once per row and scatters it across that
/// row's `m_active` support atoms. This must be BIT-FOR-BIT identical to the
/// pre-hoist algorithm (recompute the col-dot inside the support loop). Build
/// the reference diagonal here from the raw resident `(L_i, Y_i, a_phi)` with
/// the old inner-recompute structure, factor it the same way, and assert the
/// resident-built preconditioner's applied output matches it to the last bit.
#[test]
pub(crate) fn resident_scalar_jacobi_col_dot_hoist_bit_identical() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64;
    let q = 4usize;
    let p = 5usize;
    let n_atoms = 20usize;
    let m_active = 4usize; // >1 ⇒ the hoist actually folds redundant col-dots.
    let (sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, q, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;
    let k = sys.k;

    let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
        .expect("SAE structure must yield a resident operator");

    // Reference diagonal via the EXACT pre-hoist nested structure: for each
    // active support entry, recompute the col-dot inside the j loop. Same
    // additions in the same order ⇒ identical f64 bits as the hoisted form,
    // which only moves the (loop-invariant) col-dot out of the support loop.
    let mut diag_ref = Array1::<f64>::zeros(k);
    {
        let slice = diag_ref.as_slice_mut().unwrap();
        sys.penalty_diagonal_add(slice);
    }
    for a in 0..k {
        diag_ref[a] += ridge_beta;
    }
    for row in 0..resident.rows.len() {
        let rf = &resident.rows[row];
        let di = rf.di;
        if di == 0 {
            continue;
        }
        let support = &resident.a_phi[row];
        // #1033: L_i is the shared local_jac slab (was per-row rf.l).
        let l_i = &resident.local_jac[row];
        for &(beta_base, phi) in support {
            if phi == 0.0 {
                continue;
            }
            let phi2 = phi * phi;
            for j in 0..p {
                let mut col_dot = 0.0_f64;
                for r in 0..di {
                    let idx = r * p + j;
                    col_dot += l_i[idx] * rf.y[idx];
                }
                diag_ref[beta_base + j] -= phi2 * col_dot;
            }
        }
    }

    // Apply the reference diagonal directly (1/diag scaling) and the actual
    // resident-built preconditioner; compare to a tight relative tolerance. Force
    // the serial build branch to remove chunk-fold reassociation, but the col-dot
    // hoist still sums in a different order than the inner-recompute, so parity is
    // to f64 precision (rel < 1e-12), not bit-for-bit.
    let r = Array1::from_iter((0..k).map(|a| 0.4 * ((a as f64) * 0.013).cos() + 0.06));
    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("one-thread pool");
    let out_resident = one_thread.install(|| {
        JacobiPreconditioner::build_scalar_jacobi_resident(&sys, ridge_beta, &resident)
            .expect("resident scalar Jacobi")
            .apply(&r)
    });
    // The col-dot hoist computes each diagonal entry's column dot in a different
    // summation ORDER than the inner-recompute reference. f64 addition is
    // non-associative, so the two CANNOT be bit-identical — demanding `==` was an
    // over-specification. Assert genuine numerical parity at a tight relative
    // tolerance instead: a real device/CPU or hoist divergence would still fail,
    // only the unavoidable last-ULP reassociation is tolerated.
    let scale = (0..k).fold(1.0_f64, |m, a| m.max((r[a] / diag_ref[a]).abs()));
    let mut max_rel = 0.0_f64;
    for a in 0..k {
        let want = r[a] / diag_ref[a];
        max_rel = max_rel.max((out_resident[a] - want).abs() / scale);
    }
    assert!(
        max_rel < 1e-12,
        "col-dot hoist must match inner-recompute to reassociation error \
         (rel {max_rel:e})"
    );
}

/// #1017 SAE-resident scalar-Jacobi build parallelism: `build_scalar_jacobi_resident`
/// fans its per-row support sweep over rayon above `SCHUR_MATVEC_PARALLEL_ROW_MIN`,
/// accumulating worker-private length-`K` diagonal partials folded back in chunk
/// order. The point-elimination term scatters into a SHARED diagonal, so the
/// parallel build must (a) be bit-identical run-to-run and (b) reproduce the
/// serial chunk-free build up to chunk reassociation (asserted to `rel < 1e-12`,
/// NOT bit-for-bit; the serial branch is taken inside a single-thread rayon
/// worker, where `current_thread_index()` is `Some`). A diagonal drifting beyond
/// that margin would change the PCG iterate and could move the criterion ranking
/// — the #1017 determinism gate. Because (b) is tolerance-equal not bit-exact,
/// the ranking is stable only up to the reassociation margin; a near-tie winner
/// inside it can still flip (#1211).
#[test]
pub(crate) fn parallel_resident_scalar_jacobi_deterministic_and_matches_serial() {
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
    let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
        .expect("SAE structure must yield a resident operator");
    let ridge_beta = 1e-6;
    let k = sys.k;
    let r = Array1::from_iter((0..k).map(|a| 0.3 * ((a as f64) * 0.019).sin() + 0.05));

    // Two live (parallel) builds: bit-identical apply run-to-run.
    let par_a = JacobiPreconditioner::build_scalar_jacobi_resident(&sys, ridge_beta, &resident)
        .expect("resident scalar Jacobi a");
    let par_b = JacobiPreconditioner::build_scalar_jacobi_resident(&sys, ridge_beta, &resident)
        .expect("resident scalar Jacobi b");
    let out_a = par_a.apply(&r);
    let out_b = par_b.apply(&r);
    for a in 0..k {
        assert_eq!(
            out_a[a].to_bits(),
            out_b[a].to_bits(),
            "parallel resident scalar Jacobi must apply deterministically at {a}"
        );
    }

    // Serial branch: force the nested-worker gate (single-thread pool ⇒
    // `current_thread_index()` is `Some` ⇒ sequential `row = 0..n` sweep). The
    // chunk-ordered fold (`diag - Σ_chunk partial`) regroups the per-row
    // subtractions vs the serial path's `(diag - a) - b - …`, so the difference
    // is pure ULP-scale float reassociation (the SAME reassociation the generic
    // `build_scalar_jacobi`/`schur_matvec` parallel paths accept) — not a
    // numerics change; assert agreement to rel < 1e-12.
    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("one-thread pool");
    let out_serial = one_thread.install(|| {
        JacobiPreconditioner::build_scalar_jacobi_resident(&sys, ridge_beta, &resident)
            .expect("serial resident scalar Jacobi")
            .apply(&r)
    });
    let scale = out_serial
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    for a in 0..k {
        let rel = (out_a[a] - out_serial[a]).abs() / scale;
        assert!(
            rel < 1e-12,
            "parallel chunk-ordered fold must match the serial subtraction to \
             reassociation at {a}: {} vs {} (rel {rel:e})",
            out_a[a],
            out_serial[a]
        );
    }
}

/// The #1017 SAE-resident block-Jacobi builder must assemble the same
/// block-diagonal Schur preconditioner as the generic block builder, without
/// materializing each row's dense `H_tβ`. This is the block-preconditioner
/// residency gate for per-atom blocks under `BLOCK_JACOBI_MAX_BLOCK`.
#[test]
pub(crate) fn resident_block_jacobi_deterministic_and_matches_generic() {
    let n = SCHUR_MATVEC_PARALLEL_ROW_MIN + 64;
    let q = 3usize;
    let p = 6usize;
    let n_atoms = 18usize;
    let m_active = 4usize;
    let (mut sys, _a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
    let offsets: Vec<std::ops::Range<usize>> =
        (0..n_atoms).map(|atom| atom * p..(atom + 1) * p).collect();
    sys.set_block_offsets(offsets.into());
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, q, false)
        .expect("SPD per-row blocks must factor");
    let ridge_beta = 1e-6;
    let r = Array1::from_iter((0..sys.k).map(|a| 0.3 * ((a as f64) * 0.017).sin() + 0.08));

    let generic =
        JacobiPreconditioner::build_block_jacobi(&sys, &htt_factors, ridge_beta, &backend)
            .expect("generic block Jacobi must build");
    let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
        .expect("SAE structure must yield a resident operator");
    let resident_a = JacobiPreconditioner::build_block_jacobi_resident(&sys, ridge_beta, &resident)
        .expect("resident block Jacobi a");
    let resident_b = JacobiPreconditioner::build_block_jacobi_resident(&sys, ridge_beta, &resident)
        .expect("resident block Jacobi b");

    let out_generic = generic.apply(&r);
    let out_a = resident_a.apply(&r);
    let out_b = resident_b.apply(&r);
    for a in 0..sys.k {
        assert_eq!(
            out_a[a].to_bits(),
            out_b[a].to_bits(),
            "resident block Jacobi must apply deterministically at {a}"
        );
    }
    let scale = out_generic
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
    for a in 0..sys.k {
        let rel = (out_a[a] - out_generic[a]).abs() / scale;
        assert!(
            rel < 1e-10,
            "resident vs generic block Jacobi must agree at index {a}: \
             {} vs {} (rel {rel:e})",
            out_a[a],
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
        // #1033: L_i is the shared local_jac slab (was per-row rf.l).
        let l_i = &resident.local_jac[row];
        // Reconstruct the dense block G_i = L_iᵀ Y_i (p×p) from the stored
        // factors and check the factored GEMV chain against a direct G_i·g.
        let l = ArrayView2::from_shape((di, p), l_i.as_slice()).unwrap();
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
            let lrow = &l_i[r * p..r * p + p];
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
    // Storage check: the factored form keeps di·p (not p²) per row. L_i is the
    // shared local_jac slab (#1033, not re-stored in the row factor), so count it
    // from there; only Y_i is per-row in the factor.
    let factored_entries: usize = resident
        .rows
        .iter()
        .enumerate()
        .map(|(row, r)| resident.local_jac[row].len() + r.y.len())
        .sum();
    let dense_entries: usize = resident.rows.iter().filter(|r| r.di > 0).count() * p * p;
    assert!(
        factored_entries < dense_entries,
        "factored residency must store fewer entries than the dense p×p form \
             ({factored_entries} vs {dense_entries})"
    );

    // #1033 no-second-copy pin: the resident operator's L_i slab is the SAME
    // allocation as the assembler's DeviceSaePcgData.local_jac, not a per-row
    // copy. A regression that re-introduced rf.l (a verbatim copy) would fail
    // this Arc::ptr_eq even while every matvec above stayed numerically equal.
    let data = sys
        .device_sae_pcg
        .as_ref()
        .expect("structured SAE system must carry device_sae_pcg");
    assert!(
        std::sync::Arc::ptr_eq(&resident.local_jac, &data.local_jac),
        "resident operator must SHARE the assembler's local_jac slab (#1033), not copy it"
    );
}

/// #1017 preconditioner-build parallelism: `JacobiPreconditioner::build_block_jacobi`
/// — the term-block-Jacobi PCG preconditioner built once per inexact-PCG solve
/// (so O(inner-Newton-iters) times per fit) — fans its per-row reduced-Schur
/// sub-block sweep over rayon above `SCHUR_MATVEC_PARALLEL_ROW_MIN`. It must be
/// (a) DETERMINISTIC run-to-run — bit-identical regardless of thread scheduling
/// (so the preconditioner is invariant to thread SCHEDULE run-to-run); and
/// (b) numerically equal to the sequential per-row fold up to ULP-level
/// reassociation. Asserted through the applied output `P⁻¹ r` (the factored
/// block apply), which is what the PCG iterate actually consumes. Because (b) is
/// tolerance-equal not bit-for-bit with serial, the criterion ranking the
/// preconditioner feeds is stable only up to the reassociation margin and a
/// near-tie winner inside it can flip — not an exact no-move guarantee (#1211).
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
            let view = gam_linalg::faer_ndarray::FaerArrayView::new(&ref_blocks[bidx]);
            gam_linalg::faer_ndarray::FaerLlt::new(view.as_ref(), Side::Lower)
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
    let scale = ref_out
        .iter()
        .fold(0.0_f64, |m, &v| m.max(v.abs()))
        .max(1.0);
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

/// The parallel disjoint-range prefix fan-out in `CompositePenaltyOp::matvec`
/// (per-atom Kronecker smooth blocks over the K=32k manifold border) must be
/// BIT-IDENTICAL to the plain serial per-op sum. This builds a composite wide
/// enough to trip the parallel prefix (covered width ≥ `SCHUR_PROLOGUE_PARALLEL_K_MIN`,
/// ≥ 2 blocks) with a trailing dense op that overlaps every prefix index (the
/// serial tail), and asserts exact f64 agreement with an independent serial
/// reference built from `op.matvec`.
#[test]
fn composite_penalty_parallel_prefix_matches_serial_bit_exact() {
    let n_atoms = 8usize;
    let p_a = 4usize; // left Kronecker factor dim
    let p = 32usize; // identity-right width
    let block = p_a * p; // 128
    let k = n_atoms * block; // 1024 ≥ SCHUR_PROLOGUE_PARALLEL_K_MIN (512)
    assert!(
        k >= SCHUR_PROLOGUE_PARALLEL_K_MIN,
        "must trip the parallel prefix"
    );

    // Deterministic pseudo-random SPD-ish left factors and input.
    let mut state = 0x1234_5678u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };

    let mut ops: Vec<Arc<dyn BetaPenaltyOp>> = Vec::with_capacity(n_atoms + 1);
    for atom in 0..n_atoms {
        let mut a = Array2::<f64>::zeros((p_a, p_a));
        for v in a.iter_mut() {
            *v = next();
        }
        ops.push(Arc::new(IdentityRightKroneckerPenaltyOp {
            factor_a: a,
            p,
            global_offset: atom * block,
            k,
        }));
    }
    // Trailing dense op: a None-range tail that writes EVERY index, exercising
    // the "prefix-parallel then serial-tail" accumulation order.
    let mut dense = Array2::<f64>::zeros((k, k));
    for v in dense.iter_mut() {
        *v = next() * 0.01;
    }
    ops.push(Arc::new(DensePenaltyOp(dense)));

    let x: Array1<f64> = Array1::from_iter((0..k).map(|_| next()));
    let x_slice = x.as_slice().unwrap();

    // Independent serial reference: sum each op through `op.matvec` in order.
    let mut reference = vec![0.0_f64; k];
    for op in &ops {
        op.matvec(x_slice, &mut reference);
    }

    let composite = CompositePenaltyOp { k, ops };
    let mut got = vec![0.0_f64; k];
    composite.matvec(x_slice, &mut got);

    assert_eq!(
        got, reference,
        "parallel-prefix composite matvec must be bit-identical to the serial sum"
    );

    // Running it again (accumulate contract) must also match a doubled serial ref.
    let mut reference2 = reference.clone();
    for op in &composite.ops {
        op.matvec(x_slice, &mut reference2);
    }
    composite.matvec(x_slice, &mut got);
    assert_eq!(
        got, reference2,
        "second accumulating matvec must remain bit-identical to serial"
    );
}

/// The matrix-free reduced-Schur log-determinant `slq_reduced_schur_log_det`
/// (Stochastic Lanczos Quadrature on the `schur_matvec` apply, NO dense `k×k`
/// Schur formed) must agree with the exact dense evidence log|S| it replaces —
/// #1017 CPU perf: `cholesky_lower` routes the wide reduced Schur (k ≥ 128)
/// through faer's blocked LLT instead of the scalar triple loop. The blocked
/// factor must reconstruct the SAME SPD matrix (`A = L Lᵀ`) as the scalar
/// reference to a tight tolerance, be exactly lower-triangular (zero strictly
/// above the diagonal), and yield the same log-determinant — otherwise the
/// reduced solve and REML evidence that consume it would drift. Fixture width
/// 200 clears the `FAER_CHOLESKY_MIN = 128` gate so this exercises the faer
/// branch (the small direct-Schur tests below stay on the scalar path).
#[test]
fn cholesky_lower_faer_path_matches_scalar_reference_on_wide_schur() {
    let k = 200usize;
    // Well-conditioned SPD: MᵀM + k·I.
    let mut m = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            m[[i, j]] = 0.001 * (((i + 3) * (j + 1)) as f64).sin();
        }
    }
    let mut a = m.t().dot(&m);
    for i in 0..k {
        a[[i, i]] += k as f64;
    }
    // Scalar reference (pre-#1017 body), independent of the routine under test.
    let mut ref_l = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for kk in 0..j {
                sum -= ref_l[[i, kk]] * ref_l[[j, kk]];
            }
            ref_l[[i, j]] = if i == j {
                sum.sqrt()
            } else {
                sum / ref_l[[j, j]]
            };
        }
    }
    let l = cholesky_lower(&a).expect("wide SPD reduced Schur must factor");
    let mut max_factor_diff = 0.0_f64;
    for i in 0..k {
        for j in 0..k {
            if j > i {
                assert_eq!(
                    l[[i, j]],
                    0.0,
                    "faer factor must be lower-triangular at ({i},{j})"
                );
            } else {
                max_factor_diff = max_factor_diff.max((l[[i, j]] - ref_l[[i, j]]).abs());
            }
        }
    }
    // Reconstruction A ≈ L Lᵀ and log-det parity are the load-bearing invariants;
    // the raw factor entries may differ by the blocked vs scalar rounding.
    let recon = l.dot(&l.t());
    let mut max_recon = 0.0_f64;
    for i in 0..k {
        for j in 0..k {
            max_recon = max_recon.max((recon[[i, j]] - a[[i, j]]).abs());
        }
    }
    assert!(
        max_recon < 1e-8,
        "faer Cholesky must reconstruct A to 1e-8 (max |LLᵀ-A| = {max_recon})"
    );
    let logdet_faer: f64 = (0..k).map(|i| 2.0 * l[[i, i]].ln()).sum();
    let logdet_ref: f64 = (0..k).map(|i| 2.0 * ref_l[[i, i]].ln()).sum();
    assert!(
        (logdet_faer - logdet_ref).abs() < 1e-9,
        "faer vs scalar log-det mismatch: {logdet_faer} vs {logdet_ref} \
         (max factor entry diff {max_factor_diff})"
    );
}

/// Dense power-iteration reference for the top eigenvalue of an SPD matrix — a
/// self-contained oracle for [`reduced_schur_lambda_max`] that needs no eigh
/// import. Converges to `λ_max` from below; 200 steps is far more than the
/// well-separated fixture needs.
fn dense_top_eigenvalue(a: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut v = Array1::<f64>::from_elem(n, 1.0);
    let inv = v.dot(&v).sqrt().recip();
    v.mapv_inplace(|x| x * inv);
    let mut lambda = 0.0;
    for _ in 0..200 {
        let av = a.dot(&v);
        lambda = v.dot(&av);
        let norm = av.dot(&av).sqrt();
        if norm == 0.0 {
            break;
        }
        v = av / norm;
    }
    lambda
}

/// #2576 — the evidence lane's log-determinant must be PRECONDITIONED, and the
/// preconditioner must not move the value.
///
/// The reduced Schur `S = H_ββ − Σ_i H_βt(H_tt)⁻¹H_tβ` inherits `H_ββ`'s
/// diagonal spread. On the overcomplete SAE border that spread IS the atom
/// firing-count distribution — atoms occurring in a handful of rows next to
/// atoms occurring in thousands — and every shifted solve inside the rational
/// surrogate paid for it, unpreconditioned, at `√κ` convergence. This fixture
/// puts four decades of shared-block diagonal on an otherwise ordinary arrow
/// system and asserts both halves of the fix at once:
///
///   * the shared-block-diagonal tier takes strictly fewer shifted-CG
///     iterations than the identity tier (the preconditioner does something);
///   * both tiers return the SAME `log|S|` to solve accuracy (it does ONLY
///     that — the surrogate is a function of the operator, and PCG converges
///     to the same certified solve as CG).
///
/// A regression that dropped the preconditioner fails the first limb; one that
/// let it leak into the functional fails the second.
#[test]
fn evidence_logdet_preconditioner_cuts_iterations_without_moving_log_det() {
    let (n, d, k) = (24usize, 2usize, 40usize);
    let mut sys = dense_direct_system(n, d, k);
    // Four decades of shared-block diagonal spread, in the firing-count shape:
    // a few "hot" columns carrying orders of magnitude more mass than the rest.
    // The cross-block is left as the fixture built it, so the eliminated term
    // is unchanged and the spread genuinely comes from `H_ββ`.
    for column in 0..k {
        let scale = 10.0_f64.powi((column % 5) as i32);
        sys.hbb[[column, column]] = 6.0 * scale;
    }
    sys.refresh_row_hessian_fingerprint();
    let backend = CpuBatchedBlockSolver;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");

    let rows = reduced_schur_logdet_preconditioner_study(
        &sys,
        &htt_factors,
        0.0,
        &backend,
        16,
        0x2576_C0DE,
        1.0e-8,
        60,
        1.0e-10,
        50_000,
    )
    .expect("the study must run on a well-conditioned SPD arrow system");
    assert_eq!(rows.len(), 2, "study must report both tiers");

    let identity = rows
        .iter()
        .find(|row| row.preconditioner == ReducedSchurCgPreconditioner::Identity)
        .expect("identity tier");
    let scaled = rows
        .iter()
        .find(|row| row.preconditioner == ReducedSchurCgPreconditioner::SharedBlockDiagonal)
        .expect("shared-block-diagonal tier");
    eprintln!(
        "evidence log|S| preconditioner study: identity {} iters (log|S| {:.9e}), \
         shared-block diagonal {} iters (log|S| {:.9e})",
        identity.cg_iterations, identity.log_det, scaled.cg_iterations, scaled.log_det
    );

    assert!(
        scaled.cg_iterations < identity.cg_iterations,
        "the shared-block diagonal must cut the surrogate's shifted-CG work on a \
         wide-diagonal border (identity {}, scaled {})",
        identity.cg_iterations,
        scaled.cg_iterations
    );
    let gap = (scaled.log_det - identity.log_det).abs();
    assert!(
        gap <= 1.0e-6 * identity.log_det.abs().max(1.0),
        "a preconditioner steers the iteration and may not move the functional: \
         identity {:.12e} vs scaled {:.12e} (gap {gap:.3e})",
        identity.log_det,
        scaled.log_det
    );
}

/// The #2080 fixed-rational log-det surrogate on the matrix-free `schur_matvec`
/// apply (`rational_reduced_schur_log_det`, NO dense `k×k` Schur formed) must
/// agree with the exact dense evidence `log|S|` it replaces, be bit-reproducible
/// for a fixed seed (the REML outer loop differentiates a DETERMINISTIC
/// objective), and bracket the spectrum correctly via the matrix-free power
/// iteration. Companion to `slq_reduced_schur_log_det_matches_dense_evidence` —
/// the surrogate's added contract (value/gradient one functional) is exercised
/// separately by `rational_reduced_schur_directional_matches_fd_of_surrogate`.
#[test]
fn rational_reduced_schur_log_det_matches_dense_evidence() {
    let (n, d, k) = (40usize, 3usize, 80usize);
    let sys = dense_direct_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1e-6;
    let seed = 0x2080_0B0A_C0DE_u64;

    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");

    // Exact dense reduced-Schur log|S| and top eigenvalue — the O(k²) assembly
    // the matrix-free surrogate avoids, kept here only as the test oracle.
    let schur = build_dense_schur_direct(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        gam_gpu::GpuPolicy::Auto,
    )
    .expect("dense reduced Schur must build for the well-conditioned fixture");
    let l = cholesky_lower(&schur).expect("reduced Schur must be SPD");
    let exact_logdet: f64 = (0..k).map(|i| 2.0 * l[[i, i]].ln()).sum();
    let true_lambda_max = dense_top_eigenvalue(&schur);

    // Spectral bracket: power iteration on `schur_matvec` recovers λ_max
    // (Rayleigh quotient converges from below, so it never exceeds the truth).
    // The surrogate only needs a bracket good to a factor — its quadrature window
    // is padded two decades each side — so assert a factor-of-2 band rather than a
    // tight eigenvalue tolerance, which would be flaky when the top two
    // eigenvalues are close (slow power-iteration convergence).
    let lambda_max = reduced_schur_lambda_max(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        80,
        seed,
    )
    .expect("power iteration must produce a finite positive λ_max");
    assert!(
        lambda_max <= true_lambda_max * (1.0 + 1e-9),
        "power-iteration Rayleigh quotient cannot exceed the true λ_max \
         (est={lambda_max}, true={true_lambda_max})"
    );
    assert!(
        lambda_max >= 0.5 * true_lambda_max,
        "spectral-bracket λ_max must be within a factor of 2 of the truth \
         (est={lambda_max}, true={true_lambda_max})"
    );

    // Matrix-free surrogate value — never forms S.
    let (_plan, eval) = rational_reduced_schur_log_det(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        64, // num_probes
        seed,
        1e-9, // rel_tol (quadrature)
        40,   // power_iters
        1e-11,
        20_000,
    )
    .expect("rational surrogate must evaluate for the SPD fixture");
    let rel = (eval.estimate - exact_logdet).abs() / exact_logdet.abs();
    eprintln!(
        "matrix-free reduced-Schur log|S|: rational={:.6} exact={:.6} rel={:.3e} std_err={:.3e}",
        eval.estimate, exact_logdet, rel, eval.std_err
    );
    assert!(
        rel < 0.05,
        "matrix-free rational reduced-Schur log|S| rel err {rel:.3e} exceeds 5% \
         (rational={}, exact={exact_logdet})",
        eval.estimate
    );

    // Bit-reproducible for a fixed seed.
    let (_plan2, eval2) = rational_reduced_schur_log_det(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        64,
        seed,
        1e-9,
        40,
        1e-11,
        20_000,
    )
    .expect("rational surrogate must re-evaluate");
    assert_eq!(
        eval.estimate, eval2.estimate,
        "the fixed plan (fixed probes + fixed quadrature) must be bit-deterministic"
    );
}

/// `rational_reduced_schur_plan_derived` (the build-once companion): its typed
/// handoff must retain the exact certified entry evaluation that selected the
/// plan, so production does not rerun the same operator/preconditioner ladder.
/// The derived Hutch++ deflation rank must (a) leave the log|S| estimate exact
/// (deflation is an unbiased variance-reduction split, so the value cannot move
/// outside the error bar) while (b) tightening the Hutchinson std_err below the
/// bare-probe pilot when the target bar demands it. `deflation_max_rank == 0`
/// must return the bare plan (bit-identical to
/// `rational_reduced_schur_log_det`'s plan). The derived plan's frozen `Q` is
/// what the gradient contracts against, so this pins the value the criterion
/// swap will consume.
#[test]
fn rational_reduced_schur_plan_derived_deflates_to_target() {
    let (n, d, k) = (40usize, 3usize, 80usize);
    let sys = dense_direct_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1e-6;
    let seed = 0x2080_DEF1_u64;

    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let schur = build_dense_schur_direct(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        gam_gpu::GpuPolicy::Auto,
    )
    .expect("dense reduced Schur must build");
    let l = cholesky_lower(&schur).expect("reduced Schur must be SPD");
    let exact_logdet: f64 = (0..k).map(|i| 2.0 * l[[i, i]].ln()).sum();

    // Bare pilot (rank-0): the variance the deflation must beat.
    let bare = rational_reduced_schur_plan_derived(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        32,
        seed,
        1e-9,
        40,
        1e-11,
        20_000,
        0,
        4,
        0.0,
    )
    .expect("bare plan must build");
    let bare_eval = bare.entry_evaluation;
    assert!(
        (bare_eval.estimate - exact_logdet).abs() / exact_logdet.abs() < 0.05,
        "bare surrogate estimate {} must match dense {exact_logdet}",
        bare_eval.estimate
    );

    // Derived rank: an aggressive target (well under the bare std_err) forces the
    // peel to grow. The returned plan's frozen Q reduces the Hutchinson bar and
    // leaves the estimate exact.
    //
    // The rank CEILING must give the doubling ladder headroom to actually
    // certify this aggressive bar. This fixture's reduced Schur is near-scalar
    // (`hbb = 6·I`, and every row's `htbeta` block is r-independent ⇒ rank-1, so
    // the Schur correction `C = 0.65·W Wᵀ` has ‖C‖ ≈ 0.04 and κ(S) ≈ 1.008): the
    // off-diagonal `log(S/c)` mass is spread across ~40 cosine directions rather
    // than concentrated on two thin tails, so a rank-32 two-sided peel removes
    // only a fraction of the variance and cannot reach 0.1·bare. The bar is
    // reachable — `std_err → 0` monotonically as the frozen basis approaches full
    // rank (a full basis projects every probe to zero, leaving the deterministic
    // term1 = exact log|S|) — but only with a ceiling that lets the peel grow
    // past 32. Use `k`: the ladder still STOPS at the first rank that certifies,
    // so on a genuinely wide-κ operator it returns a low-rank Q; here it peels
    // deeper because the fixture demands it. This keeps the aggressive 0.1× bar
    // (a real quality contract) rather than weakening it to whatever rank-32
    // happens to achieve on a poorly-conditioned-for-deflation fixture.
    let target_rel = 0.1 * bare_eval.std_err / (exact_logdet.abs() + 1.0);
    let derived = rational_reduced_schur_plan_derived(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        32,
        seed,
        1e-9,
        40,
        1e-11,
        20_000,
        k, // deflation_max_rank: resource ceiling with headroom to certify 0.1×bare
        6, // subspace_iters
        target_rel,
    )
    .expect("derived plan must build");
    let derived_eval = derived.entry_evaluation;
    eprintln!(
        "derived-rank plan: est={:.6} exact={:.6} bare_std_err={:.3e} derived_std_err={:.3e}",
        derived_eval.estimate, exact_logdet, bare_eval.std_err, derived_eval.std_err
    );
    assert!(
        (derived_eval.estimate - exact_logdet).abs() / exact_logdet.abs() < 0.05,
        "deflation must not bias the estimate: derived={} exact={exact_logdet}",
        derived_eval.estimate
    );
    assert!(
        derived_eval.std_err < bare_eval.std_err,
        "Hutch++ deflation must reduce the std_err below the bare probe pilot \
         (bare={:.3e}, derived={:.3e})",
        bare_eval.std_err,
        derived_eval.std_err
    );

    // The rank ceiling is resource admission, not a license to consume an
    // under-certified stochastic criterion. A zero requested bar cannot be met
    // by one deflated direction with a finite probe block, so the plan must
    // refuse instead of returning the deepest attempted Q.
    let under_certified = rational_reduced_schur_plan_derived(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        32,
        seed,
        1e-9,
        40,
        1e-11,
        20_000,
        1,
        2,
        0.0,
    );
    assert!(
        under_certified.is_none(),
        "derived surrogate must refuse when its rank ceiling is exhausted before \
         the requested Hutchinson error bar is certified"
    );
}

/// Dense reference `tr(S⁻¹)` from the lower-Cholesky factor `S = L Lᵀ`:
/// `tr(S⁻¹) = tr(L⁻ᵀ L⁻¹) = ‖L⁻¹‖_F²`, with each `L⁻¹` column solved by forward
/// substitution (`L y = e_c`). Self-contained oracle for the matrix-free
/// `tr(S⁻¹·M)` estimator, no eigensolver needed.
fn dense_trace_inverse(l: &Array2<f64>) -> f64 {
    let k = l.nrows();
    let mut acc = 0.0;
    for c in 0..k {
        let mut y = vec![0.0_f64; k];
        for i in 0..k {
            let mut s = if i == c { 1.0 } else { 0.0 };
            for j in 0..i {
                s -= l[[i, j]] * y[j];
            }
            y[i] = s / l[[i, i]];
        }
        acc += y.iter().map(|v| v * v).sum::<f64>();
    }
    acc
}

/// The matrix-free `tr(S⁻¹·M)` Hutchinson estimator (#2080 general umbrella):
/// the `S⁻¹ v_j` bundle (`reduced_schur_inverse_probe_solves`, `t = 0` CG on
/// `schur_matvec`) contracted against a channel matvec. `M = S` is the exact
/// plumbing check (`tr(S⁻¹ S) = k` with ZERO variance, since `(S⁻¹v)ᵀ(Sv) =
/// ‖v‖² = k`), and `M = I` exercises the genuine Hutchinson estimate of
/// `tr(S⁻¹)` against the dense oracle. Also pins determinism.
#[test]
fn hutchinson_reduced_schur_inverse_trace_matches_dense() {
    let (n, d, k) = (40usize, 3usize, 80usize);
    let sys = dense_direct_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1e-6;
    let seed = 0x2080_51_7A_u64;

    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let schur = build_dense_schur_direct(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        gam_gpu::GpuPolicy::Auto,
    )
    .expect("dense reduced Schur must build");
    let l = cholesky_lower(&schur).expect("reduced Schur must be SPD");
    let exact_tr_inv = dense_trace_inverse(&l);

    // Fixed probe set (reuse the surrogate plan's Rademacher probes).
    let plan = RationalLogdetPlan::build(k, 64, seed, 1e-3, 1e3, 1e-9).expect("plan");
    let (sinv, _) = reduced_schur_inverse_probe_solves(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        &plan.probes,
        None,
        1e-12,
        50_000,
    )
    .expect("S⁻¹ v_j bundle must solve");

    // M = S ⇒ tr(S⁻¹S) = k, variance-free (a plumbing + solve-accuracy gate).
    let tr_sinv_s = hutchinson_reduced_schur_inverse_trace(&plan.probes, &sinv, &|v| {
        let x = v.to_owned();
        let mut out = Array1::<f64>::zeros(k);
        schur_matvec(&sys, &htt_factors, ridge_beta, &x, &mut out, &backend, None);
        out
    })
    .expect("tr(S⁻¹S) estimate");
    let rel_s = (tr_sinv_s - k as f64).abs() / k as f64;
    assert!(
        rel_s < 1e-5,
        "tr(S⁻¹S) must equal k to solve accuracy: got {tr_sinv_s} vs k={k} (rel {rel_s:.3e})"
    );

    // M = I ⇒ tr(S⁻¹) against the dense forward-substitution oracle.
    let tr_sinv_i = hutchinson_reduced_schur_inverse_trace(&plan.probes, &sinv, &|v| v.to_owned())
        .expect("tr(S⁻¹) estimate");
    let rel_i = (tr_sinv_i - exact_tr_inv).abs() / exact_tr_inv.abs().max(1e-12);
    eprintln!("tr(S⁻¹): est={tr_sinv_i:.6} exact={exact_tr_inv:.6} rel={rel_i:.3e}");
    assert!(
        rel_i < 0.15,
        "matrix-free tr(S⁻¹) rel err {rel_i:.3e} exceeds 15% (est {tr_sinv_i} vs exact {exact_tr_inv})"
    );

    // Determinism: the fixed probe set + deterministic CG reproduce bit-for-bit.
    let sinv2 = reduced_schur_inverse_probe_solves(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        &plan.probes,
        None,
        1e-12,
        50_000,
    )
    .expect("S⁻¹ v_j bundle must re-solve");
    let (sinv2, _) = sinv2;
    let tr2 = hutchinson_reduced_schur_inverse_trace(&plan.probes, &sinv2, &|v| v.to_owned())
        .expect("tr(S⁻¹) re-estimate");
    assert_eq!(tr_sinv_i, tr2, "tr(S⁻¹) estimator must be bit-reproducible");
}

/// Dense SPD solve `S⁻¹ rhs` from the lower-Cholesky factor `S = L Lᵀ`: forward
/// substitution `L y = rhs` then back substitution `Lᵀ x = y`. Oracle for the
/// matrix-free single-rhs [`reduced_schur_inverse_apply`].
fn dense_spd_solve_from_lower(l: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    let k = l.nrows();
    let mut y = vec![0.0_f64; k];
    for i in 0..k {
        let mut s = rhs[i];
        for j in 0..i {
            s -= l[[i, j]] * y[j];
        }
        y[i] = s / l[[i, i]];
    }
    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut s = y[i];
        for j in i + 1..k {
            s -= l[[j, i]] * x[j];
        }
        x[i] = s / l[[i, i]];
    }
    Array1::from_vec(x)
}

/// The matrix-free single-rhs reduced-Schur solve
/// [`reduced_schur_inverse_apply`] (the base primitive for the selected-inverse
/// gradient channels whose `S⁻¹` argument is per-call, not the fixed probe
/// bundle) must reproduce the dense `S⁻¹ rhs` to solve accuracy and be
/// bit-reproducible for a fixed rhs.
#[test]
fn reduced_schur_inverse_apply_matches_dense_solve() {
    let (n, d, k) = (40usize, 3usize, 80usize);
    let sys = dense_direct_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1e-6;

    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");
    let schur = build_dense_schur_direct(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        gam_gpu::GpuPolicy::Auto,
    )
    .expect("dense reduced Schur must build");
    let l = cholesky_lower(&schur).expect("reduced Schur must be SPD");

    // Fixed Rademacher rhs (deterministic, no eigensolver needed).
    let mut state = 0x2080_A951_C0DE_u64;
    let rhs = Array1::<f64>::from_shape_fn(k, |_| {
        if gam_linalg::utils::splitmix64(&mut state) & 1 == 1 {
            1.0
        } else {
            -1.0
        }
    });
    let dense_x = dense_spd_solve_from_lower(&l, &rhs);

    let (mf_x, _) = reduced_schur_inverse_apply(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        &rhs,
        None,
        1e-12,
        50_000,
    )
    .expect("matrix-free S⁻¹ rhs must solve");
    let err = (&mf_x - &dense_x).mapv(|x| x * x).sum().sqrt();
    let scale = dense_x.mapv(|x| x * x).sum().sqrt().max(1e-12);
    let rel = err / scale;
    eprintln!("matrix-free S⁻¹ rhs: rel err {rel:.3e}");
    assert!(
        rel < 1e-6,
        "matrix-free S⁻¹ rhs must match the dense L Lᵀ solve to CG accuracy (rel {rel:.3e})"
    );

    // Bit-reproducible for a fixed rhs (the REML gradient lane requires it).
    let (mf_x2, _) = reduced_schur_inverse_apply(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        &rhs,
        None,
        1e-12,
        50_000,
    )
    .expect("matrix-free S⁻¹ rhs must re-solve");
    assert_eq!(mf_x, mf_x2, "single-rhs S⁻¹ solve must be bit-reproducible");

    // Warm-start slot: seeding with the exact solution converges to it (the CRN
    // reuse the surrogate lane does across the ρ walk cannot move the answer, only
    // cut iterations).
    let (mf_warm, _) = reduced_schur_inverse_apply(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
        None,
        None,
        &rhs,
        Some(&dense_x),
        1e-12,
        50_000,
    )
    .expect("warm-started S⁻¹ rhs must solve");
    let warm_rel = (&mf_warm - &dense_x).mapv(|x| x * x).sum().sqrt() / scale;
    assert!(
        warm_rel < 1e-6,
        "warm-starting from the exact solution must return it (rel {warm_rel:.3e})"
    );
}

/// #2230 production seam: the full-arrow matrix-free operator and arbitrary-RHS
/// inverse used by the SAE exact-stationarity IFT solve must represent the same
/// undamped bordered Hessian as the dense factor cache. This pins both halves:
/// `Bv` (including reconstruction of `H_betabeta` from the reduced Schur) and
/// `B^-1 r` (matrix-free beta CG plus exact row back-substitution).
#[test]
fn matrix_free_full_arrow_apply_and_inverse_match_dense_cache() {
    let (n, d, k) = (24usize, 3usize, 48usize);
    let sys = dense_direct_system(n, d, k);
    let options = ArrowSolveOptions::direct().with_positive_definite_evidence();
    let (_, _, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
        .expect("undamped dense oracle factorization");

    let t_len = cache.delta_t_len();
    let vector_t =
        Array1::<f64>::from_shape_fn(t_len, |index| 0.2 * ((index as f64 + 1.0) * 0.37).sin());
    let vector_beta =
        Array1::<f64>::from_shape_fn(k, |index| 0.15 * ((index as f64 + 2.0) * 0.23).cos());
    let (dense_t, dense_beta) =
        arrow_operator_apply(&sys, 0.0, 0.0, vector_t.view(), vector_beta.view());
    let (matrix_free_t, matrix_free_beta) =
        matrix_free_arrow_operator_apply(&sys, &cache, vector_t.view(), vector_beta.view())
            .expect("matrix-free full-arrow apply");
    let apply_error = (&matrix_free_t - &dense_t)
        .mapv(|value| value * value)
        .sum()
        + (&matrix_free_beta - &dense_beta)
            .mapv(|value| value * value)
            .sum();
    let apply_scale =
        dense_t.mapv(|value| value * value).sum() + dense_beta.mapv(|value| value * value).sum();
    assert!(
        apply_error.sqrt() <= 1.0e-11 * apply_scale.sqrt().max(1.0),
        "matrix-free Bv must match the dense assembled operator: rel={:.3e}",
        apply_error.sqrt() / apply_scale.sqrt().max(1.0)
    );

    let rhs_t =
        Array1::<f64>::from_shape_fn(t_len, |index| 0.1 * ((index as f64 + 3.0) * 0.41).cos());
    let rhs_beta =
        Array1::<f64>::from_shape_fn(k, |index| 0.12 * ((index as f64 + 4.0) * 0.19).sin());
    let (dense_solved_t, dense_solved_beta) = cache
        .full_inverse_apply(rhs_t.view(), rhs_beta.view())
        .expect("dense full-arrow inverse");
    let (matrix_free_solved_t, matrix_free_solved_beta, _) = matrix_free_arrow_inverse_apply(
        &sys,
        &cache,
        rhs_t.view(),
        rhs_beta.view(),
        1.0e-12,
        50_000,
    )
    .expect("matrix-free full-arrow inverse");
    let inverse_error = (&matrix_free_solved_t - &dense_solved_t)
        .mapv(|value| value * value)
        .sum()
        + (&matrix_free_solved_beta - &dense_solved_beta)
            .mapv(|value| value * value)
            .sum();
    let inverse_scale = dense_solved_t.mapv(|value| value * value).sum()
        + dense_solved_beta.mapv(|value| value * value).sum();
    assert!(
        inverse_error.sqrt() <= 1.0e-7 * inverse_scale.sqrt().max(1.0),
        "matrix-free B^-1 r must match the dense cache solve to CG accuracy: rel={:.3e}",
        inverse_error.sqrt() / inverse_scale.sqrt().max(1.0)
    );
}

/// #1017 resident-context parity: [`ReducedSchurOperator`] on the CPU lane
/// (`gpu_matvec == None`) must be BIT-IDENTICAL to the inline `schur_matvec`
/// closure it replaces across the rational-logdet / SLQ ladder. The whole point
/// of the widened-lifetime operator is that staging it once and reusing it across
/// every shifted solve cannot move a single ULP versus the per-solve-closure form.
#[test]
fn reduced_schur_operator_cpu_lane_is_bit_identical_to_schur_matvec() {
    let (n, d, k) = (32usize, 3usize, 64usize);
    let sys = dense_direct_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1e-6;

    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");

    // The CPU operator: no device matvec attached, so every apply routes through
    // `schur_matvec` with the shared (here `None`) residency.
    let op = ReducedSchurOperator::new(&sys, &htt_factors, ridge_beta, &backend, None);

    // Several deterministic Rademacher probes — the operator's `apply` /
    // `apply_owned` must reproduce a direct `schur_matvec` call byte-for-byte.
    let mut state = 0x1017_0FEE_C0DE_u64;
    for _ in 0..5 {
        let v = Array1::<f64>::from_shape_fn(k, |_| {
            if gam_linalg::utils::splitmix64(&mut state) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        });
        let mut expected = Array1::<f64>::zeros(k);
        schur_matvec(
            &sys,
            &htt_factors,
            ridge_beta,
            &v,
            &mut expected,
            &backend,
            None,
        );

        let got_view = op.apply(v.view());
        assert_eq!(
            got_view, expected,
            "ReducedSchurOperator::apply must be bit-identical to schur_matvec"
        );
        let got_owned = op.apply_owned(&v);
        assert_eq!(
            got_owned, expected,
            "ReducedSchurOperator::apply_owned must be bit-identical to schur_matvec"
        );
    }
}

/// #1017 resident-context lifecycle: a device operator attached via
/// [`ReducedSchurOperator::with_gpu_matvec`] is staged ONCE and every shifted
/// solve of a ladder reuses it — the "upload once per criterion evaluation"
/// contract, verified with a mock [`GpuSchurMatvec`] that counts its applies. The
/// operator must (a) route ALL applies to the single attached device matvec
/// (never fall back to the CPU `schur_matvec`), and (b) never rebuild it per
/// apply — the call count equals the number of ladder applies exactly.
#[test]
fn reduced_schur_operator_device_matvec_is_uploaded_once_and_reused() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let (n, d, k) = (8usize, 2usize, 16usize);
    let sys = dense_direct_system(n, d, k);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1e-6;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks must factor");

    // Mock device operator: an identity apply `out = x` that counts every call.
    // Building the Arc ONCE models the "upload once"; the counter proves reuse.
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_c = Arc::clone(&calls);
    let gpu: GpuSchurMatvec = Arc::new(move |x: &Array1<f64>, out: &mut Array1<f64>| {
        calls_c.fetch_add(1, Ordering::Relaxed);
        out.assign(x);
    });

    // Attach the resident device operator to a single operator instance.
    let op = ReducedSchurOperator::new(&sys, &htt_factors, ridge_beta, &backend, None)
        .with_gpu_matvec(Some(&gpu));

    // A ladder of applies (mimicking the shift ladder's repeated matvecs). Each
    // must be served by the attached device operator, not the CPU path: the
    // identity output proves the device lane was taken (a real `schur_matvec` on
    // this SPD system would NOT return the input unchanged).
    const APPLIES: usize = 11;
    for i in 0..APPLIES {
        let v = Array1::<f64>::from_elem(k, (i as f64) + 1.0);
        let got = op.apply(v.view());
        assert_eq!(
            got, v,
            "the attached device matvec (identity) must serve every apply"
        );
    }
    assert_eq!(
        calls.load(Ordering::Relaxed),
        APPLIES,
        "the resident device operator must be reused across every ladder apply \
         (uploaded once, not rebuilt per solve)"
    );

    // With NO device operator the SAME operator config falls back to the CPU
    // `schur_matvec` — the byte-identical default lane.
    let cpu_op = ReducedSchurOperator::new(&sys, &htt_factors, ridge_beta, &backend, None);
    let probe = Array1::<f64>::from_elem(k, 1.0);
    let mut expected = Array1::<f64>::zeros(k);
    schur_matvec(
        &sys,
        &htt_factors,
        ridge_beta,
        &probe,
        &mut expected,
        &backend,
        None,
    );
    assert_eq!(
        cpu_op.apply(probe.view()),
        expected,
        "gpu_matvec=None must route to schur_matvec (byte-identical CPU fallback)"
    );
    // And the device apply-count is untouched by the CPU-lane operator.
    assert_eq!(
        calls.load(Ordering::Relaxed),
        APPLIES,
        "the CPU-lane operator must not touch the device matvec"
    );
}

// ---------------------------------------------------------------------------
// Co-visibility cluster preconditioner (Kushal & Agarwal visibility-based
// preconditioning). At real over-complete SAE widths the co-firing graph is a
// single giant connected component, so the component-partition cluster tier
// exceeds the size cap and degrades to scalar Jacobi — the scaling ceiling.
// The bounded co-visibility partition splits that component into strongly-
// co-firing clusters whose dense factors condition the cross-atom coupling
// scalar Jacobi cannot see.
// ---------------------------------------------------------------------------

/// The co-visibility cluster-size cap is DERIVED from the per-factor memory
/// budget, not asserted as a bare number: `b_max = ⌊√(budget/8)⌋`. Pin it equal
/// to the legacy `CLUSTER_JACOBI_MAX_CLUSTER` scalar-fallback ceiling so the
/// bounded co-visibility partition and the component-partition builders agree.
#[test]
pub(crate) fn covisibility_cap_is_derived_from_factor_budget() {
    let b = ((CLUSTER_SCHUR_FACTOR_BYTES_BUDGET / 8) as f64)
        .sqrt()
        .floor() as usize;
    assert_eq!(
        covisibility_cluster_max_cols(),
        b,
        "cap must equal ⌊√(budget/8)⌋"
    );
    assert_eq!(
        covisibility_cluster_max_cols(),
        CLUSTER_JACOBI_MAX_CLUSTER,
        "derived cap must coincide with the legacy scalar-fallback ceiling (REML-neutral)"
    );
    let cap = covisibility_cluster_max_cols() as u128;
    assert!(8 * cap * cap <= CLUSTER_SCHUR_FACTOR_BYTES_BUDGET);
    assert!(8 * (cap + 1) * (cap + 1) > CLUSTER_SCHUR_FACTOR_BYTES_BUDGET);
}

/// Apply the point-elimination correction `C = Σ_i H_tβ(i)ᵀ (H_tt(i))⁻¹ H_tβ(i)`
/// to a β-vector: `C v = Σ_i H_tβ(i)ᵀ (H_tt(i))⁻¹ (H_tβ(i) v)`. `C` is the PSD
/// operator subtracted from `H_ββ` to form the reduced Schur `S = H_ββ − C`.
fn apply_correction(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &CpuBatchedBlockSolver,
    v: &Array1<f64>,
) -> Array1<f64> {
    let mut cv = Array1::<f64>::zeros(sys.k);
    for (i, row) in sys.rows.iter().enumerate() {
        let hv = row.htbeta.dot(v); // d
        let solved = backend.solve_block_vector(htt_factors.factor(i), hv.view()); // d
        cv += &row.htbeta.t().dot(&solved); // k
    }
    cv
}

/// Power-iterate the correction operator `C` for its top eigenvalue λ_max(C).
/// Used to place `H_ββ = λ_max(C)·(1+ε)·I` so the reduced Schur `S = H_ββ − C`
/// is GUARANTEED SPD (`S ⪰ ε·λ_max(C)·I ≻ 0`) with a KNOWN condition number
/// `κ(S) ≈ (1+ε)/ε` (since the near-low-rank `C` has `λ_min(C) ≈ 0`): the whole
/// point of the fixture is a genuinely ill-conditioned S, not one flattered by a
/// dominant penalty.
fn correction_lambda_max(
    sys: &ArrowSchurSystem,
    htt_factors: &ArrowFactorSlab,
    backend: &CpuBatchedBlockSolver,
    iters: usize,
) -> f64 {
    let k = sys.k;
    // Deterministic non-degenerate seed.
    let mut v: Array1<f64> = Array1::from_iter((0..k).map(|j| ((j + 1) as f64 * 0.7).sin() + 0.3));
    let mut nrm = v.dot(&v).sqrt();
    if nrm > 0.0 {
        v /= nrm;
    }
    let mut lambda = 0.0;
    for _ in 0..iters {
        let cv = apply_correction(sys, htt_factors, backend, &v);
        lambda = v.dot(&cv); // Rayleigh quotient
        nrm = cv.dot(&cv).sqrt();
        if nrm == 0.0 {
            break;
        }
        v = cv / nrm;
    }
    // One more Rayleigh at the converged vector.
    let cv = apply_correction(sys, htt_factors, backend, &v);
    lambda.max(v.dot(&cv))
}

/// Build an over-complete co-activating dictionary with a planted co-firing
/// GROUP structure and REPRESENTATIVE numerics: overlapping ambient subspaces
/// and heavy-tailed within-group co-firing.
///
/// `n_groups` groups, each of `blocks_per_group` β-blocks of width `block_width`
/// (so a group spans `blocks_per_group*block_width` columns). Every group has
/// `rows_per_group` rows whose `H_tβ` fires on ALL of that group's columns
/// (strong intra-group co-firing → the co-firing graph clusters by group), and
/// consecutive groups are stitched by ONE weak bridge row each (co-firing the
/// last block of group `g` with the first block of group `g+1`), making the whole
/// co-firing graph a SINGLE connected component — the regime where the
/// component-partition cluster tier exceeds the size cap and degrades to scalar
/// Jacobi.
///
/// Representativeness (the reviewer's regime — "co-activating atoms with
/// overlapping ambient subspaces"): within a group every row's `H_tβ` is a
/// rank-≤`d` outer product of a latent-axis profile with a SHARED few-mode column
/// profile `ψ(local, g)` (overlapping subspaces), scaled by a heavy-tailed
/// per-row weight `1/(j+1)` (heavy-tailed co-firing). All group rows therefore lie
/// in the SAME low-dimensional column subspace, so the within-group correction
/// `C_g` is strongly near-rank-deficient — exactly the coupling a dense per-group
/// Cholesky conditions and scalar diagonal cannot. `H_ββ` is left zero here and set
/// by the caller from `λ_max(C)` so S is SPD with a controlled condition number.
fn covisibility_planted_group_system(
    n_groups: usize,
    blocks_per_group: usize,
    block_width: usize,
    d: usize,
    rows_per_group: usize,
    strong: f64,
    bridge: f64,
) -> (ArrowSchurSystem, usize) {
    let num_blocks = n_groups * blocks_per_group;
    let group_width = blocks_per_group * block_width;
    let k = num_blocks * block_width;
    let n = n_groups * rows_per_group + n_groups.saturating_sub(1);
    let mut sys = ArrowSchurSystem::new(n, d, k);
    let group_rows_end = n_groups * rows_per_group;
    // Column co-firing profile for group `g`, row `j`: a DOMINANT mode shared by
    // every row of the group (the overlapping ambient subspace all the group's
    // atoms load on — the near-rank-deficient direction that makes the within-
    // group Schur ill-conditioned) plus a per-row mode (so `C_g` carries several
    // comparable co-firing modes rather than a single rank-1 direction, i.e. the
    // scalar diagonal must resolve each one). `g` keys the frequencies so groups
    // occupy different column subspaces.
    let psi = |local: usize, g: usize, j: usize| -> f64 {
        let x = (local as f64) / (group_width as f64);
        let shared = (std::f64::consts::PI * (1.0 + g as f64) * x).sin();
        let per_row =
            (std::f64::consts::PI * (2.0 + g as f64 + j as f64) * x + 0.3 * j as f64).sin();
        shared + 0.7 * per_row
    };
    for (i, row) in sys.rows.iter_mut().enumerate() {
        for r in 0..d {
            for c in 0..d {
                row.htt[[r, c]] = if r == c { 4.0 + (i % 3) as f64 } else { 0.15 };
            }
            row.gt[r] = 0.05 * ((i + r + 1) as f64).sin();
        }
        if i < group_rows_end {
            let g = i / rows_per_group;
            let j = i % rows_per_group; // 0-based row within the group
            let col0 = g * group_width;
            // Mild heavy-tailed per-row weight (representative of heavy-tailed
            // co-firing; slow enough that several within-group modes stay above the
            // PCG tolerance and must be resolved). Latent-axis profile shared across
            // rows so the group's rows lie in the same low-dim column subspace.
            let weight = strong / (1.0 + 0.5 * j as f64).sqrt();
            for r in 0..d {
                let latent = (0.6 * (r as f64) + 0.4 * (g as f64) + 1.0).cos();
                for local in 0..group_width {
                    row.htbeta[[r, col0 + local]] = weight * latent * psi(local, g, j);
                }
            }
        } else {
            let b = i - group_rows_end;
            let last_block_col0 = (b * group_width) + (group_width - block_width);
            let next_block_col0 = (b + 1) * group_width;
            for r in 0..d {
                for local in 0..block_width {
                    row.htbeta[[r, last_block_col0 + local]] =
                        bridge * ((local + r + 1) as f64).cos();
                    row.htbeta[[r, next_block_col0 + local]] =
                        bridge * ((local + r + 2) as f64).sin();
                }
            }
        }
    }
    for r in 0..k {
        sys.gb[r] = 0.02 * ((r + 1) as f64).cos();
    }
    let mut offsets: Vec<Range<usize>> = Vec::with_capacity(num_blocks);
    for blk in 0..num_blocks {
        offsets.push((blk * block_width)..((blk + 1) * block_width));
    }
    sys.set_block_offsets(std::sync::Arc::from(offsets.into_boxed_slice()));
    sys.refresh_row_hessian_fingerprint();
    (sys, group_width)
}

/// The bounded co-visibility partition recovers the planted co-firing groups and
/// its cluster-Jacobi preconditioner drives the reduced-Schur PCG to the SAME
/// solution as scalar Jacobi in materially fewer iterations. At these widths the
/// co-firing graph is one giant component, so the component-partition
/// `ClusterJacobi` exceeds the cap and degrades to the scalar reciprocal
/// diagonal; the co-visibility partition conditions the strong, near-rank-
/// deficient within-group coupling scalar Jacobi cannot see. The reported gap is
/// a MEASUREMENT; the assertion is a modest, structurally-derived bound.
#[test]
pub(crate) fn covisibility_partition_recovers_groups_and_beats_scalar_jacobi() {
    use std::ops::Range;
    // Each group = blocks_per_group × block_width = the derived cap, so a group
    // exactly fills a cluster and the bounded partition separates groups cleanly.
    let cap = covisibility_cluster_max_cols();
    let block_width = 64usize;
    let blocks_per_group = cap / block_width; // 8
    let n_groups = 4usize;
    let d = 4usize;
    let rows_per_group = 6usize;
    let (mut sys, group_width) = covisibility_planted_group_system(
        n_groups,
        blocks_per_group,
        block_width,
        d,
        rows_per_group,
        0.9,
        0.02,
    );
    let k = sys.k;
    assert!(
        k > cap,
        "fixture must exceed the cluster cap (k={k}, cap={cap})"
    );

    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1e-8;
    let htt_factors = backend
        .factor_blocks(&sys.rows, 0.0, d, false)
        .expect("SPD per-row blocks factor");

    // Place H_ββ = λ_max(C)·(1+ε)·I so S = H_ββ − C is SPD with κ(S) ≈ (1+ε)/ε
    // — a genuinely ill-conditioned reduced Schur (ε small), not one flattered by
    // a dominant penalty. ε = 1/32 ⇒ κ(S) ≈ 33.
    let lambda_max_c = correction_lambda_max(&sys, &htt_factors, &backend, 40);
    assert!(lambda_max_c.is_finite() && lambda_max_c > 0.0);
    let epsilon = 1.0 / 32.0;
    let hbb_diag = lambda_max_c * (1.0 + epsilon);
    for r in 0..k {
        sys.hbb[[r, r]] = hbb_diag;
    }

    // The co-firing graph must be a SINGLE connected component (the ceiling
    // regime), and the bounded co-visibility partition must recover the planted
    // groups: n_groups clusters, each exactly one group's columns.
    let graph = BetaCouplingGraph::build_from_system(&sys);
    assert_eq!(
        graph.component_partition().len(),
        1,
        "bridged dictionary must be one connected co-firing component"
    );
    let covis = graph.covisibility_cluster_partition(&sys.block_offsets, cap);
    assert_eq!(
        covis.len(),
        n_groups,
        "co-visibility partition must recover {n_groups} planted groups, got {}",
        covis.len()
    );
    for (ci, cluster) in covis.iter().enumerate() {
        let cols: usize = cluster.iter().map(|&b| sys.block_offsets[b].len()).sum();
        assert_eq!(cols, group_width, "cluster {ci} must be one planted group");
        let g0 = cluster[0] / blocks_per_group;
        assert!(
            cluster.iter().all(|&b| b / blocks_per_group == g0),
            "cluster {ci} must not straddle planted groups"
        );
    }

    let rhs: Array1<f64> =
        Array1::from_iter((0..k).map(|j| 0.3 * ((j + 1) as f64).sin() + 0.1 * (j as f64).cos()));
    let pcg = ArrowPcgOptions {
        max_iterations: 8 * k,
        relative_tolerance: 1e-10,
    };
    let trust = ArrowTrustRegionOptions {
        radius: 1.0e12,
        steihaug_relative_tolerance: 1e-10,
        max_iterations: 8 * k,
    };

    // (a) Scalar Jacobi baseline: clear block_offsets so the Jacobi build takes
    // the per-column scalar-diagonal path — the ceiling the cluster tier collapses
    // to at these widths.
    let (scalar_sol, scalar_diag) = {
        let mut bare = sys.clone();
        bare.set_block_offsets(std::sync::Arc::from([] as [Range<usize>; 0]));
        let bare_factors = backend
            .factor_blocks(&bare.rows, 0.0, bare.d, false)
            .expect("bare factors");
        let jac = JacobiPreconditioner::from_arrow_schur(
            &bare,
            &bare_factors,
            ridge_beta,
            &backend,
            None,
        )
        .expect("scalar Jacobi build");
        run_pcg_with_preconditioner(
            &bare,
            &bare_factors,
            ridge_beta,
            &rhs,
            |r| jac.apply(r),
            &pcg,
            &trust,
            &backend,
            None,
            None,
            None,
        )
        .expect("scalar-Jacobi PCG")
    };

    // (b) Co-visibility cluster-Jacobi.
    let covis_pc = ClusterJacobiPreconditioner::from_arrow_schur_covisibility(
        &sys,
        &htt_factors,
        ridge_beta,
        &backend,
    )
    .expect("co-visibility cluster build");
    let (covis_sol, covis_diag) = run_pcg_with_preconditioner(
        &sys,
        &htt_factors,
        ridge_beta,
        &rhs,
        |r| covis_pc.apply(r),
        &pcg,
        &trust,
        &backend,
        None,
        None,
        None,
    )
    .expect("co-visibility PCG");

    let ratio = scalar_diag.iterations as f64 / (covis_diag.iterations.max(1) as f64);
    eprintln!(
        "[covisibility] k={k} clusters={n_groups} kappa~{:.0} lambda_max_C={:.3e} | \
         scalar_jacobi_iters={} (conv={}) covis_iters={} (conv={}) ratio={:.1}x",
        (1.0 + epsilon) / epsilon,
        lambda_max_c,
        scalar_diag.iterations,
        matches!(scalar_diag.stopping_reason, PcgStopReason::Converged),
        covis_diag.iterations,
        matches!(covis_diag.stopping_reason, PcgStopReason::Converged),
        ratio
    );

    // Correctness: both preconditioners solve the SAME reduced system, so their
    // converged solutions must agree tightly (the preconditioner steers the CG
    // path, not the fixed point — REML-neutral).
    assert!(
        matches!(scalar_diag.stopping_reason, PcgStopReason::Converged),
        "scalar-Jacobi baseline must converge (iters={}, rel_resid={:e})",
        scalar_diag.iterations,
        scalar_diag.final_relative_residual
    );
    assert!(
        matches!(covis_diag.stopping_reason, PcgStopReason::Converged),
        "co-visibility cluster-Jacobi must converge (iters={}, rel_resid={:e})",
        covis_diag.iterations,
        covis_diag.final_relative_residual
    );
    let mut max_abs = 0.0f64;
    let mut ref_norm = 0.0f64;
    for j in 0..k {
        max_abs = max_abs.max((scalar_sol[j] - covis_sol[j]).abs());
        ref_norm = ref_norm.max(scalar_sol[j].abs());
    }
    let rel = if ref_norm > 0.0 {
        max_abs / ref_norm
    } else {
        max_abs
    };
    assert!(
        rel < 1e-6,
        "covis and scalar solves must agree (same S); rel diff {rel:e}"
    );

    // Regression — modest, structurally-derived bound (NOT tuned to the measured
    // gap). The co-visibility clusters are the planted groups, so cluster-Jacobi
    // inverts each group's near-rank-deficient within-group Schur exactly; PCG
    // then only resolves the weak inter-group bridge coupling. Scalar Jacobi keeps
    // none of the within-group coupling. With `n_groups` such groups each carrying
    // strong within-group coupling scalar cannot precondition, removing them cuts
    // the CG iteration count by at least a factor of 2 (a conservative floor on the
    // group/bridge mode-count ratio, which is ≈ rows_per_group·n_groups/(n_groups−1)).
    // The actual measured factor is printed above.
    assert!(
        covis_diag.iterations < scalar_diag.iterations,
        "co-visibility must strictly reduce PCG iterations vs scalar Jacobi: covis={} scalar={}",
        covis_diag.iterations,
        scalar_diag.iterations
    );
    assert!(
        covis_diag.iterations * 2 <= scalar_diag.iterations,
        "co-visibility must at least halve PCG iterations vs scalar Jacobi (derived floor): \
         covis={} scalar={}",
        covis_diag.iterations,
        scalar_diag.iterations
    );
}

/// Lower forward-substitution solve `Lx=b`, then upper (Lᵀ) back-substitution
/// `Lᵀy=x` — a minimal, self-contained `(LLᵀ)⁻¹b` solve for these tests, so they
/// exercise the factor `factor_dense_reduced_schur` returns without depending
/// on any other crate's triangular-solve helper.
fn solve_via_lower_cholesky(factor: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = factor.nrows();
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = b[i];
        for j in 0..i {
            acc -= factor[[i, j]] * y[j];
        }
        y[i] = acc / factor[[i, i]];
    }
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut acc = y[i];
        for j in i + 1..n {
            acc -= factor[[j, i]] * x[j];
        }
        x[i] = acc / factor[[i, i]];
    }
    x
}

/// #2015 — `factor_dense_reduced_schur`'s internal Jacobi/Van der Sluis
/// equilibration (design: issue 2015 comment 4949898801) must return a factor
/// that reconstructs the CALLER'S ORIGINAL matrix exactly (`L·Lᵀ = S`), not
/// some scaled proxy — the whole point of the fix is that every existing
/// consumer keeps reading real, original-unit values.
#[test]
fn factor_dense_reduced_schur_reconstructs_original_illconditioned_matrix_2015() {
    let n = 6usize;
    // Planted SPD matrix with a genuine ~1e4 diagonal spread (mirrors the
    // measured real-data output column-norm spread): a diagonal core plus a
    // small, symmetric off-diagonal coupling that keeps it non-trivially
    // dense without threatening positive-definiteness (Gershgorin: each row's
    // off-diagonal mass is a small fraction of its own diagonal entry).
    let diag_scale = [1.0e4_f64, 1.0e2, 1.0, 1.0e-2, 1.0, 1.0e-4];
    let mut schur = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        schur[[i, i]] = diag_scale[i];
    }
    // Coupling `c · min(diag_i, diag_j)`: each off-diagonal entry is bounded by
    // `c` times the SMALLER of its row's/column's own diagonal, so for ANY row
    // the sum of its (n-1) off-diagonal magnitudes is at most
    // `c · (n-1) · diag_i` (since `min(diag_i, diag_j) ≤ diag_i`) — strictly
    // less than `diag_i` for `c · (n-1) < 1` (here `c=1e-3`, `n-1=5`). This
    // guarantees strict diagonal dominance, hence genuine positive-definiteness,
    // for EVERY row regardless of how extreme the diagonal spread is — unlike a
    // `sqrt(diag_i·diag_j)`-scaled coupling, which can violate dominance at the
    // smallest-diagonal row.
    let coupling_fraction = 1.0e-3_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let coupling = coupling_fraction * diag_scale[i].min(diag_scale[j]);
            schur[[i, j]] = coupling;
            schur[[j, i]] = coupling;
        }
    }

    let DenseReducedSchurFactorization {
        factor,
        conditioned_schur: floored,
        beta_conditioning: _,
    } = factor_dense_reduced_schur(&schur, ReducedSchurPolicy::StrictNewton)
        .expect("planted matrix is PD");
    assert!(
        floored.is_none(),
        "a genuinely PD matrix must not need the spectral floor"
    );

    let reconstructed = factor.dot(&factor.t());
    let mut max_abs_diff = 0.0_f64;
    let mut max_scale = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            max_abs_diff = max_abs_diff.max((reconstructed[[i, j]] - schur[[i, j]]).abs());
            max_scale = max_scale.max(schur[[i, j]].abs());
        }
    }
    let relative = max_abs_diff / max_scale.max(1.0);
    assert!(
        relative < 1e-9,
        "L·Lᵀ must reconstruct the ORIGINAL (unequilibrated) matrix; relative diff {relative:e}"
    );

    // Solve S x = b for a planted x, both via the returned factor and via a
    // reference solve on the SAME matrix Cholesky-factored directly (no
    // equilibration) — the well-conditioned columns here make the direct path
    // trustworthy as a reference. Agreement must be tight (roundoff-level, not
    // bit-identical: the two factors are computed via different arithmetic
    // paths), matching the requested "not bit-identical, roundoff differs"
    // tolerance of 1e-10 relative.
    let x_true = Array1::from_vec(vec![1.0, -2.0, 0.5, 3.0, -1.5, 2.0]);
    let b = schur.dot(&x_true);
    let x_via_equilibrated_factor = solve_via_lower_cholesky(&factor, &b);
    let reference_factor =
        cholesky_lower(&schur).expect("planted matrix is PD for direct Cholesky too");
    let x_via_direct_factor = solve_via_lower_cholesky(&reference_factor, &b);

    let mut max_abs = 0.0_f64;
    let mut ref_norm = 0.0_f64;
    for i in 0..n {
        max_abs = max_abs.max((x_via_equilibrated_factor[i] - x_via_direct_factor[i]).abs());
        ref_norm = ref_norm.max(x_via_direct_factor[i].abs());
    }
    let relative_solve_diff = max_abs / ref_norm.max(1.0);
    assert!(
        relative_solve_diff < 1e-10,
        "the equilibrated-then-reconstructed factor's solve must agree with the direct \
         Cholesky solve to roundoff, got relative diff {relative_solve_diff:e}"
    );
    // And both must actually recover the planted x.
    for i in 0..n {
        assert!(
            (x_via_equilibrated_factor[i] - x_true[i]).abs() < 1e-6,
            "solved x[{i}]={} must recover planted x_true[{i}]={}",
            x_via_equilibrated_factor[i],
            x_true[i]
        );
    }
}

/// #2308 — an evidence β-null is pinned to unit stiffness in ORIGINAL β
/// coordinates whether it is slightly negative (Cholesky refusal) or slightly
/// positive (successful but sub-floor Cholesky). It contributes `log 1 = 0`,
/// and the explicit mask records exactly the direction the inverse must drop.
#[test]
fn evidence_beta_schur_boundary_has_unit_logdet_and_authoritative_mask_2308() {
    for collapsed in [-1.0e-12_f64, 1.0e-12_f64] {
        let schur = array![[4.0_f64, 0.0], [0.0, collapsed]];
        let evidence = factor_dense_reduced_schur(
            &schur,
            ReducedSchurPolicy::EvidenceUnitDeflation {
                relative_floor: SPECTRAL_DEFLATION_REL_FLOOR,
                refuse_resolved_indefinite: false,
            },
        )
        .expect("evidence unit deflation");
        let spectrum = evidence
            .beta_conditioning
            .as_ref()
            .expect("sub-floor β direction must carry metadata");
        assert_eq!(
            &*spectrum.conditioning,
            &[
                BetaSchurSpectralConditioning::UnitDeflated,
                BetaSchurSpectralConditioning::Raw,
            ]
        );
        assert_eq!(
            spectrum.raw_evals[0].is_sign_negative(),
            collapsed.is_sign_negative(),
            "the raw eigenspectrum must preserve which side of zero the boundary lies on"
        );
        let raw_scale = spectrum
            .raw_evals
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        assert!(
            spectrum.raw_evals[0].abs() < SPECTRAL_DEFLATION_REL_FLOOR * raw_scale,
            "the authoritative mask must identify a genuinely sub-floor raw direction"
        );

        // A symmetric eigensolver is backward stable, not an exact scalar
        // oracle. Certify each returned raw eigenpair against the source Schur
        // operator with a gamma_n bound scaled by ||S||_inf; demanding 1e-18
        // agreement with a diagonal literal is below the binary64 backward
        // error of this O(1)-scaled problem.
        let schur_norm_inf = (0..schur.nrows())
            .map(|row| schur.row(row).iter().map(|value| value.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        let operation_count = schur.ncols().saturating_mul(2).saturating_add(2);
        let accumulated = operation_count as f64 * (0.5 * f64::EPSILON);
        assert!(accumulated < 1.0);
        let gamma = accumulated / (1.0 - accumulated);
        for eigen_index in 0..spectrum.raw_evals.len() {
            let eigenvalue = spectrum.raw_evals[eigen_index];
            let eigenvector = spectrum.evecs.column(eigen_index);
            let eigenvector_norm = eigenvector
                .iter()
                .fold(0.0_f64, |norm, value| norm.max(value.abs()));
            let mut residual_norm = 0.0_f64;
            for row in 0..schur.nrows() {
                let mut action = 0.0_f64;
                for column in 0..schur.ncols() {
                    action += schur[[row, column]] * eigenvector[column];
                }
                residual_norm = residual_norm.max((action - eigenvalue * eigenvector[row]).abs());
            }
            let backward_error_allowance =
                gamma * (schur_norm_inf + eigenvalue.abs()) * eigenvector_norm;
            assert!(
                residual_norm <= backward_error_allowance,
                "raw eigenpair {eigen_index} residual {residual_norm:e} exceeds its scale-derived backward-error allowance {backward_error_allowance:e}"
            );
        }
        assert_eq!(spectrum.cond_evals[0], 1.0);
        assert_eq!(spectrum.cond_evals[1], 4.0);

        let log_det = (0..2)
            .map(|axis| 2.0 * evidence.factor[[axis, axis]].ln())
            .sum::<f64>();
        assert_abs_diff_eq!(log_det, 4.0_f64.ln(), epsilon = 2e-14);
        let conditioned = evidence
            .conditioned_schur
            .as_ref()
            .expect("boundary operator was conditioned");
        assert_abs_diff_eq!(conditioned[[0, 0]], 4.0, epsilon = 1e-14);
        assert_abs_diff_eq!(conditioned[[1, 1]], 1.0, epsilon = 1e-14);
    }
}

/// #2308 — in the full-rank interior evidence performs no conditioning and the
/// ordinary log-determinant remains `log|S|`. Newton Tikhonov remains a separate
/// policy with its own boundary value.
#[test]
fn evidence_beta_schur_interior_is_raw_and_newton_boundary_is_tikhonov_2308() {
    let interior = array![[4.0_f64, 0.0], [0.0, 2.0]];
    let evidence = factor_dense_reduced_schur(
        &interior,
        ReducedSchurPolicy::EvidenceUnitDeflation {
            relative_floor: SPECTRAL_DEFLATION_REL_FLOOR,
            refuse_resolved_indefinite: false,
        },
    )
    .expect("interior evidence factor");
    assert!(evidence.beta_conditioning.is_none());
    assert!(evidence.conditioned_schur.is_none());
    let evidence_log_det = (0..2)
        .map(|axis| 2.0 * evidence.factor[[axis, axis]].ln())
        .sum::<f64>();
    assert_abs_diff_eq!(evidence_log_det, 8.0_f64.ln(), epsilon = 2e-14);

    let boundary = array![[4.0_f64, 0.0], [0.0, -1.0e-12]];
    let newton = factor_dense_reduced_schur(
        &boundary,
        ReducedSchurPolicy::NewtonTikhonov {
            relative_floor: SPECTRAL_DEFLATION_REL_FLOOR,
        },
    )
    .expect("Newton Tikhonov factor");
    assert!(newton.beta_conditioning.is_none());
    let newton_conditioned = newton
        .conditioned_schur
        .as_ref()
        .expect("boundary Newton operator was Tikhonov-conditioned");
    let newton_log_det = (0..2)
        .map(|axis| 2.0 * newton.factor[[axis, axis]].ln())
        .sum::<f64>();
    let expected = (newton_conditioned[[0, 0]] * newton_conditioned[[1, 1]]
        - newton_conditioned[[0, 1]] * newton_conditioned[[1, 0]])
    .ln();
    assert_abs_diff_eq!(newton_log_det, expected, epsilon = 2e-12);
    assert!((newton_log_det - 4.0_f64.ln()).abs() > 1.0);
}

/// #2308 — the public cache seam always rebuilds the same undamped evidence
/// operator, so changing the Newton ridge history cannot change its value,
/// mask, or inverse. This exercises the metadata propagation rather than only
/// the reduced-factor helper.
#[test]
fn evidence_cache_boundary_is_invariant_to_newton_damping_history_2308() {
    let mut sys = ArrowSchurSystem::new(0, 0, 2);
    sys.hbb = array![[4.0_f64, 0.0], [0.0, -1.0e-12]];
    sys.gb = Array1::<f64>::zeros(2);
    let options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_evidence_unit_deflation(SPECTRAL_DEFLATION_REL_FLOOR);

    let (_, _, ridge_zero) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
        .expect("ridge-zero step and evidence cache");
    let (_, _, damped_step) = solve_arrow_newton_step_with_options(&sys, 0.0, 1.0e-3, &options)
        .expect("damped step and undamped evidence cache");

    for cache in [&ridge_zero, &damped_step] {
        assert_abs_diff_eq!(
            cache.arrow_log_det().expect("evidence logdet"),
            4.0_f64.ln(),
            epsilon = 2e-14
        );
        let spectrum = cache
            .beta_schur_conditioning
            .as_ref()
            .expect("β-null metadata reached cache");
        assert_eq!(
            &*spectrum.conditioning,
            &[
                BetaSchurSpectralConditioning::UnitDeflated,
                BetaSchurSpectralConditioning::Raw,
            ]
        );
        let null_rhs = array![0.0_f64, 1.0];
        let solved = cache
            .schur_inverse_apply(null_rhs.view())
            .expect("ordinary inverse consumes evidence mask");
        assert!(solved.iter().all(|value| value.abs() < 1e-12));
    }
    assert_eq!(
        ridge_zero.arrow_log_det().unwrap().to_bits(),
        damped_step.arrow_log_det().unwrap().to_bits()
    );
}

/// #2576: the EXACT reduced-Schur diagonal, built from the SAE residency, must
/// equal the operator's own diagonal — the claim
/// [`resident_schur_elimination_diagonal`] rests on.
///
/// The issue's history twice concluded that an exact `diag(S)` was unaffordable,
/// because the generic route materializes `H_tβ^(i)` against `K` basis vectors.
/// That is true of a generic cross-block and false of this one: the SAE row
/// factors as `H_tβ^(i) = L_i P_i`, so
/// `diag(S_i)[base_s + c] = φ_s²·Σ_r L_i[r,c]·Y_i[r,c]` reads straight off slabs
/// the residency already staged. This gate holds that closed form to the
/// operator itself, probed column by column with `S·e_g` — if the closed form
/// and the operator ever disagree, the preconditioner is scaling by a diagonal
/// that is not the one the iteration sees.
///
/// Size-independent (both arms are exact identities, not tolerances on a fit).
///
/// Second arm — `n_atoms = 4, m_active = 5` — forces the support generator to
/// hand the SAME base twice within one row with DIFFERENT `φ`. The projector's
/// coefficient there is `φ_a + φ_b`, so the diagonal carries `(φ_a + φ_b)²`;
/// accumulating `φ_a² + φ_b²` instead would price a different projector than the
/// matvec applies, and only a duplicate-base fixture can tell the two apart.
#[test]
pub(crate) fn resident_schur_elimination_diagonal_matches_operator_diagonal_2576() {
    for (n_atoms, m_active, arm) in [(32usize, 5usize, "distinct bases"), (4, 5, "duplicate base")] {
        let n = 48usize;
        let q = 4usize;
        let p = 6usize;
        let (sys, a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
        let k = sys.k;
        let backend = CpuBatchedBlockSolver;
        let htt_factors = backend
            .factor_blocks(&sys.rows, 0.0, q, false)
            .expect("SPD per-row blocks must factor");
        let ridge_beta = 1e-6;
        let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
            .expect("SAE structure must yield a resident operator");

        // NON-VACUITY for the duplicate arm: assert the fixture really does
        // repeat a base inside a row, or the arm proves nothing about the
        // combine-before-square rule.
        if arm == "duplicate base" {
            let repeated = a_phi.iter().any(|support| {
                support
                    .iter()
                    .enumerate()
                    .any(|(i, (base, _))| support[..i].iter().any(|(seen, _)| seen == base))
            });
            assert!(
                repeated,
                "#2576 fixture must repeat a support base within a row for this arm to bite"
            );
        }

        // The closed form under test.
        let elimination = resident_schur_elimination_diagonal(&resident, k)
            .expect("residency describes this system");
        let mut closed_form = sys.shared_block_diagonal();
        for (value, &eliminated) in closed_form.iter_mut().zip(elimination.iter()) {
            *value += ridge_beta;
            *value -= eliminated;
        }

        // The operator's own diagonal, probed column by column: diag(S)[g] =
        // e_gᵀ S e_g. This is the O(k) build the closed form exists to avoid, so
        // it is exactly the right independent witness at fixture scale.
        let mut probed = Array1::<f64>::zeros(k);
        let mut e_g = Array1::<f64>::zeros(k);
        let mut column = Array1::<f64>::zeros(k);
        for g in 0..k {
            e_g.fill(0.0);
            e_g[g] = 1.0;
            column.fill(0.0);
            schur_matvec(
                &sys,
                &htt_factors,
                ridge_beta,
                &e_g,
                &mut column,
                &backend,
                None,
            );
            probed[g] = column[g];
        }

        let scale = probed.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1.0);
        for g in 0..k {
            let rel = (closed_form[g] - probed[g]).abs() / scale;
            assert!(
                rel < 1e-12,
                "#2576 [{arm}] exact Schur diagonal must equal the operator diagonal at {g}: \
                 closed form {} vs probed {} (rel {rel:e})",
                closed_form[g],
                probed[g]
            );
        }

        // The eliminated term is PSD, so the exact diagonal is a STRICT
        // improvement on the shared-block diagonal it replaces: same positivity,
        // strictly smaller, and it must actually move (a zero elimination would
        // mean the fixture has no cross-block and the gate is vacuous).
        let shared = sys.shared_block_diagonal();
        let mut moved = false;
        for g in 0..k {
            assert!(
                elimination[g] >= -1e-12 * scale,
                "#2576 eliminated term must be PSD on the diagonal at {g}: {}",
                elimination[g]
            );
            assert!(
                closed_form[g] > 0.0,
                "#2576 exact Schur diagonal must stay positive at {g}: {}",
                closed_form[g]
            );
            if elimination[g] > 1e-9 * (shared[g].abs() + 1.0) {
                moved = true;
            }
        }
        assert!(
            moved,
            "#2576 [{arm}] fixture must carry a nonzero point-elimination term, \
             or the exact diagonal is vacuously the shared-block one"
        );

        // Determinism: the fixed-order row accumulation is bit-identical.
        let again = resident_schur_elimination_diagonal(&resident, k).expect("rebuild");
        for g in 0..k {
            assert_eq!(
                elimination[g].to_bits(),
                again[g].to_bits(),
                "#2576 exact Schur diagonal must be bit-identical run-to-run at {g}"
            );
        }
    }
}

/// #2576: how much of the border diagonal the point-elimination term actually
/// removes — measured against BOTH a synthetic and a data-accumulated `H_ββ`,
/// because that choice, not the SAE operator, decides the answer.
///
/// # Why this test exists
///
/// The exact reduced-Schur diagonal is affordable here (see
/// `resident_schur_elimination_diagonal`) and was measured NOT to cut shifted-CG
/// iterations: 5189 against the shared block's 5138, identity's 29236. I first
/// explained that by a cancellation argument — `diag(H_ββ)` and the eliminated
/// term are both `Σ_{i ∋ s} φ_{i,s}²`-weighted row sums, so the firing-count
/// spread should divide out and leave a uniform rescaling, which CG cannot see.
///
/// **That explanation was wrong and this test is what refuted it.** The measured
/// ratio spread is ~9x, not ~1x. The real cause on that fixture is MAGNITUDE:
/// the eliminated term is a fraction of a percent of the border diagonal, so
/// `diag(S)` is a ~0.2% perturbation of `diag(H_ββ)` and no iteration can
/// notice.
///
/// And that is a property of the FIXTURE. `sae_structured_system` installs a
/// synthetic uniform `hbb = k + 4`, unrelated to the cross-block it is being
/// compared against. On the real lane `H_ββ` is data-accumulated from the SAME
/// rows and the SAME `φ²` weights as the elimination, so the two are
/// commensurate by construction. Arm B builds that case, and it is the arm that
/// speaks to production.
///
/// This test reports rather than rules: the quantity that governs the
/// preconditioner is `1 − ratio` (how far `diag(S)` is from a uniform rescaling
/// of `diag(H_ββ)`), and the guard fires exactly when the elimination becomes
/// big enough that the iteration comparison could change.
#[test]
pub(crate) fn elimination_share_of_the_border_diagonal_is_fixture_dependent_2576() {
    let (n, q, p, n_atoms, m_active) = (64usize, 3usize, 6usize, 24usize, 4usize);
    let backend = CpuBatchedBlockSolver;
    let ridge_beta = 1e-6;

    // Arm A: the fixture as built — synthetic uniform shared block.
    // Arm B: `H_ββ` accumulated from the design the cross-block actually uses,
    //        `Σ_i c_{i,g}²` with `c` the row's projector coefficient on column
    //        `g`. This is the shape the production border has.
    for arm in ["synthetic uniform H_bb", "data-accumulated H_bb"] {
        let (mut sys, a_phi, _jac) = sae_structured_system(n, q, p, n_atoms, m_active);
        let k = sys.k;
        if arm == "data-accumulated H_bb" {
            let mut gram = vec![0.0_f64; k];
            for support in a_phi.iter() {
                // Combine equal bases BEFORE squaring: the projector coefficient
                // on a column is the sum over support entries carrying that base.
                let mut combined: Vec<(usize, f64)> = Vec::new();
                for &(base, phi) in support.iter() {
                    match combined.iter_mut().find(|(seen, _)| *seen == base) {
                        Some(entry) => entry.1 += phi,
                        None => combined.push((base, phi)),
                    }
                }
                for &(base, phi) in combined.iter() {
                    for value in gram[base..base + p].iter_mut() {
                        *value += phi * phi;
                    }
                }
            }
            sys.hbb = Array2::<f64>::zeros((k, k));
            for g in 0..k {
                // A small ridge keeps the synthetic border SPD where an atom is
                // unreached; it is not a tuned quantity, just non-singularity.
                sys.hbb[[g, g]] = gram[g] + 1e-3;
            }
            sys.refresh_row_hessian_fingerprint();
        }

        let htt_factors = backend
            .factor_blocks(&sys.rows, 0.0, q, false)
            .expect("SPD per-row blocks must factor");
        let resident = SaeResidentReducedSchur::build(&sys, &htt_factors, &backend)
            .expect("SAE structure must yield a resident operator");
        let elimination = resident_schur_elimination_diagonal(&resident, k)
            .expect("residency describes this system");
        let shared = sys.shared_block_diagonal();

        let mut ratios: Vec<f64> = Vec::new();
        for g in 0..k {
            let denominator = shared[g] + ridge_beta;
            if elimination[g] > 0.0 && denominator > 0.0 {
                ratios.push(elimination[g] / denominator);
            }
        }
        assert!(
            ratios.len() >= k / 2,
            "#2576 [{arm}]: the fixture must reach most border columns \
             (reached {} of {k})",
            ratios.len()
        );
        let lo = ratios.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = ratios.iter().copied().fold(0.0_f64, f64::max);
        // The quantity that governs the ITERATION: `diag(S) = diag(H_ββ)·(1 − ratio)`,
        // and CG is invariant to a uniform scale, so only the VARIATION of
        // `1 − ratio` can change anything.
        let rescale_variation = (1.0 - lo) / (1.0 - hi);
        eprintln!(
            "#2576 [{arm}] over {} reached columns: elimination/shared min {lo:.6e} \
             max {hi:.6e} (spread {:.3}x) | diag(S)/diag(H_bb) in [{:.6}, {:.6}], \
             non-uniformity {rescale_variation:.6}x",
            ratios.len(),
            hi / lo,
            1.0 - hi,
            1.0 - lo,
        );

        // The ITERATION comparison, on both arms. The ratio above predicts what
        // this should show; running it here is what keeps the #2576 verdict from
        // resting on the synthetic fixture alone. Same operator, same frozen
        // plan, same probes and nodes — only the preconditioner varies.
        match reduced_schur_logdet_preconditioner_study(
            &sys,
            &htt_factors,
            ridge_beta,
            &backend,
            16,
            0x2576_5CD1,
            1.0e-8,
            60,
            1.0e-10,
            50_000,
        ) {
            Some(rows) => {
                for row in &rows {
                    eprintln!(
                        "#2576 [{arm}] study {:<22?}: {:>7} cg iters  log|S| {:+.9e}",
                        row.preconditioner, row.cg_iterations, row.log_det
                    );
                }
                // A preconditioner steers the iteration and may not move the
                // functional; if a tier disagrees, its iteration count is
                // measuring a different problem and is not comparable.
                let reference = rows[0].log_det;
                for row in &rows {
                    let gap = (row.log_det - reference).abs();
                    assert!(
                        gap <= 1.0e-6 * reference.abs().max(1.0),
                        "#2576 [{arm}]: tier {:?} moved log|S|: {reference:.12e} vs {:.12e}",
                        row.preconditioner,
                        row.log_det
                    );
                }
            }
            None => eprintln!("#2576 [{arm}] study: REFUSED (spectral bracket or plan build)"),
        }

        // The guard, on the quantity that would flip the conclusion. It is not a
        // claim that the exact diagonal never helps — it is the trigger to
        // re-measure iterations when the elimination stops being a rounding
        // correction to the border diagonal.
        assert!(
            hi < 0.5,
            "#2576 [{arm}]: the point-elimination term now removes up to {:.1}% of the \
             border diagonal. `diag(S)` is no longer a small perturbation of \
             `diag(H_ββ)`, so the iteration comparison that retired the exact \
             diagonal (5189 vs 5138) must be re-taken through \
             `reduced_schur_logdet_preconditioner_study` before that verdict is \
             relied on.",
            hi * 100.0
        );
    }
}

/// #2731 — [`CoupledCarrierPenaltyOp`] is EXACTLY the operator its eigen
/// expansion is, on every surface the solver reads it through.
///
/// The barrier's Gauss–Newton β curvature is `Σ_{a,b} C[a,b]·v_a v_bᵀ`. It used
/// to ship as `Σ_r λ_r w_r w_rᵀ` with `w_r = Σ_a e_r[a] v_a`, one
/// `SparseRankOnePenaltyOp` per eigenvector — mathematically the same thing, and
/// the reason it was replaced is cost, not correctness (`ne²·‖v‖₀` to build,
/// `ne·‖∪ supp v‖₀` to store and apply). This test builds that expansion by hand
/// from the SAME `C` and carriers and requires the factored operator to
/// reproduce it through `matvec`, `gradient`, `diagonal`, `block` and
/// `to_dense`, so the replacement is pinned as an identity rather than as an
/// intention.
///
/// The fixture is built to be able to FAIL:
///
/// * three carriers over four blocks, with SHARED support — carriers 0 and 1
///   both touch block 1, carriers 1 and 2 both touch block 2. A factored form
///   that forgot the cross terms, or a `diagonal` that missed the overlap of two
///   carriers on one block, agrees with the expansion only when the supports are
///   disjoint. The non-vacuity assertion below pins that they are not.
/// * `C` carries a NEGATIVE eigenvalue. The operator applies `C` as handed over
///   and must not quietly clamp it: the barrier's own PSD majorization
///   (`|M|`) happens upstream, and an operator that silently projected here
///   would hide a caller that forgot to.
/// * the carrier runs are misaligned across carriers (different starts and
///   different lengths), so the run-overlap arithmetic is exercised rather than
///   short-circuited by identical layouts.
#[test]
fn coupled_carrier_penalty_op_equals_its_rank_one_expansion_2731() {
    use ndarray::Array1;

    let k = 24_usize;
    // (start, values) runs; deliberately different widths and starts.
    let carriers: Vec<Vec<(usize, Vec<f64>)>> = vec![
        vec![
            (0, vec![0.5, -1.25, 0.75]),
            (6, vec![2.0, 0.25, -0.5, 1.5]),
        ],
        vec![
            (6, vec![-0.75, 1.0, 0.5, -2.25]),
            (13, vec![0.25, -1.5, 3.0]),
        ],
        vec![(13, vec![1.75, 0.5, -0.25]), (18, vec![-1.0, 0.125])],
    ];
    let ne = carriers.len();
    assert_eq!(
        ne, 3,
        "the expansion below builds one rank-1 per eigenvector, so the carrier \
         count and the coupling's dimension must agree"
    );
    // Symmetric, deliberately indefinite: eigenvalues of this 3x3 are not all
    // positive (the trace is 6.0 and the determinant is negative).
    let coupling = array![[2.0, -1.5, 0.75], [-1.5, 1.0, 2.5], [0.75, 2.5, 3.0]];
    assert_eq!(coupling.nrows(), ne);
    let (eigenvalues, eigenvectors) = {
        use gam_linalg::faer_ndarray::FaerEigh;
        coupling
            .eigh(faer::Side::Lower)
            .expect("3x3 symmetric eigendecomposition")
    };
    assert!(
        eigenvalues.iter().any(|&lam| lam < -1.0e-9),
        "fixture must exercise an INDEFINITE coupling: {eigenvalues:?}"
    );

    // Dense carriers, then the historical expansion `Σ_r λ_r w_r w_rᵀ`.
    let dense_carriers: Vec<Array1<f64>> = carriers
        .iter()
        .map(|runs| {
            let mut v = Array1::<f64>::zeros(k);
            for (start, values) in runs {
                for (i, &value) in values.iter().enumerate() {
                    v[start + i] += value;
                }
            }
            v
        })
        .collect();
    let shared: Vec<usize> = (0..k)
        .filter(|&i| dense_carriers.iter().filter(|v| v[i] != 0.0).count() >= 2)
        .collect();
    assert!(
        shared.len() >= 4,
        "fixture must have carriers sharing support, else the cross terms are \
         untested: shared indices {shared:?}"
    );
    let mut expansion = Array2::<f64>::zeros((k, k));
    for (r, &lam) in eigenvalues.iter().enumerate() {
        let mut w = Array1::<f64>::zeros(k);
        for (a, v) in dense_carriers.iter().enumerate() {
            w = w + eigenvectors[[a, r]] * v;
        }
        for i in 0..k {
            for j in 0..k {
                expansion[[i, j]] += lam * w[i] * w[j];
            }
        }
    }

    let op = CoupledCarrierPenaltyOp {
        k,
        coupling: coupling.clone(),
        carriers,
    };
    let scale = expansion.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(scale > 1.0, "fixture operator must be non-trivial: {scale}");
    let tolerance = 1.0e-12 * (1.0 + scale);

    // to_dense
    let dense = op.to_dense();
    for i in 0..k {
        for j in 0..k {
            assert!(
                (dense[[i, j]] - expansion[[i, j]]).abs() <= tolerance,
                "to_dense mismatch at ({i},{j}): {} vs {}",
                dense[[i, j]],
                expansion[[i, j]]
            );
        }
    }

    // matvec and gradient (both are `P·vector`), against the expansion.
    let x: Vec<f64> = (0..k).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
    let mut y_op = vec![0.0_f64; k];
    let mut y_grad = vec![0.0_f64; k];
    op.matvec(&x, &mut y_op);
    op.gradient(&x, &mut y_grad);
    for i in 0..k {
        let reference: f64 = (0..k).map(|j| expansion[[i, j]] * x[j]).sum();
        assert!(
            (y_op[i] - reference).abs() <= tolerance,
            "matvec mismatch at {i}: {} vs {reference}",
            y_op[i]
        );
        assert!(
            (y_grad[i] - reference).abs() <= tolerance,
            "gradient mismatch at {i}: {} vs {reference}",
            y_grad[i]
        );
    }

    // diagonal
    let mut diag = vec![0.0_f64; k];
    op.diagonal(&mut diag);
    for i in 0..k {
        assert!(
            (diag[i] - expansion[[i, i]]).abs() <= tolerance,
            "diagonal mismatch at {i}: {} vs {}",
            diag[i],
            expansion[[i, i]]
        );
    }

    // block: ranges chosen so one straddles two carriers' shared run.
    let offsets = [0..6, 6..13, 13..18, 18..k];
    for (id, range) in offsets.iter().enumerate() {
        let b = range.end - range.start;
        let mut blk = Array2::<f64>::zeros((b, b));
        op.block(BetaBlockId(id), &offsets, &mut blk);
        for i in 0..b {
            for j in 0..b {
                let reference = expansion[[range.start + i, range.start + j]];
                assert!(
                    (blk[[i, j]] - reference).abs() <= tolerance,
                    "block {id} mismatch at ({i},{j}): {} vs {reference}",
                    blk[[i, j]]
                );
            }
        }
    }

    // The `ne = 1` case is the historical rank-1 operator, so it must still be
    // exactly `scale·v vᵀ` — the deleted `SparseRankOnePenaltyOp`'s contract.
    let rank_one = CoupledCarrierPenaltyOp {
        k: 4,
        coupling: array![[2.5]],
        carriers: vec![vec![(1, vec![3.0, -1.0])]],
    };
    let expected = array![
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 22.5, -7.5, 0.0],
        [0.0, -7.5, 2.5, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ];
    let rank_one_dense = rank_one.to_dense();
    for i in 0..4 {
        for j in 0..4 {
            assert_abs_diff_eq!(rank_one_dense[[i, j]], expected[[i, j]], epsilon = 1.0e-12);
        }
    }
}

/// #2515 — THE EXACT-OBSERVED-INFORMATION POLICY USES THE SHARED TYPED
/// B-METRIC/CLAMP VERDICT, while the majorizer policy unit-deflates its PSD
/// numerical nulls. Both arms are asserted here, because
/// either alone is satisfiable by a wrong implementation: a policy that refuses
/// everything passes the refusal arm, and the historical one-sided predicate
/// passes the pin arm.
///
/// The one-sided predicate is `value < deflate_floor`, which admits EVERY
/// negative eigenvalue however large into the "numerically null" class. For the
/// Gauss--Newton majorizer that is correct — `B` is PSD by construction, so a
/// negative eigenvalue there can only be rounding on a direction that is null
/// anyway. For the exact observed information it prices a saddle direction as the
/// rho-independent null `log 1 = 0` and inverts it at `1`, and it does so
/// SILENTLY: the conditioned factor is PD and its log-determinant is finite, so
/// nothing downstream can tell.
///
/// `-7.997610e-3` is not a fabricated magnitude. It is the reduced-Schur
/// eigenvalue measured on #2712's certified deflated anchor at
/// `log lambda_smooth = -1.05` (gam-sae's
/// `zz_attribute_the_broken_ladder_rung_2515`), where the one-sided predicate
/// pinned it to `+1` while the dense route classified the same direction as #2336
/// clamp-attributable curvature and priced it at its basin — the two complete
/// outer gradients then `1.009` RELATIVE apart.
#[test]
fn evidence_classification_prices_a_clamp_basin_before_refusing_a_saddle_2515() {
    let majorizer = ReducedSchurPolicy::EvidenceUnitDeflation {
        relative_floor: SPECTRAL_DEFLATION_REL_FLOOR,
        refuse_resolved_indefinite: false,
    };
    let exact_a = ReducedSchurPolicy::EvidenceUnitDeflation {
        relative_floor: SPECTRAL_DEFLATION_REL_FLOOR,
        refuse_resolved_indefinite: true,
    };

    // A RESOLVED negative direction: relative magnitude 2.0e-3, five decades
    // outside the 1.0e-8 null band.
    let resolved_negative = array![[4.0_f64, 0.0], [0.0, -7.997_610e-3]];
    let pinned = factor_dense_reduced_schur(&resolved_negative, majorizer)
        .expect("the majorizer policy conditions a negative direction rather than refusing");
    let spectrum = pinned
        .beta_conditioning
        .as_ref()
        .expect("the majorizer policy records the direction it pinned");
    let pinned_indices: Vec<usize> = (0..spectrum.conditioning.len())
        .filter(|&index| spectrum.conditioning[index].is_unit_deflated())
        .collect();
    assert_eq!(
        pinned_indices.len(),
        1,
        "exactly the negative direction is pinned under the majorizer policy"
    );
    assert_abs_diff_eq!(
        spectrum.cond_evals[pinned_indices[0]],
        1.0,
        epsilon = 0.0
    );
    assert!(
        spectrum.raw_evals[pinned_indices[0]] < 0.0,
        "the pinned direction is the NEGATIVE one, not the 4.0"
    );

    let basin_geometry = ExactAReducedClassification {
        majorizer_metric: array![[4.0_f64, 0.0], [0.0, 4.0]],
        clamp_metric: array![[0.0_f64, 0.0], [0.0, 2.0e-2]],
    };
    let basin = factor_dense_reduced_schur_with_exact_a(
        &resolved_negative,
        exact_a,
        Some(&basin_geometry),
    )
    .expect(
        "#2515: raw negative exact-A curvature wholly explained by the bounded ARD clamp \
         is a basin, not a saddle",
    );
    let reconstructed_basin = basin.factor.dot(&basin.factor.t());
    let basin_spectrum = basin
        .beta_conditioning
        .as_ref()
        .expect("#2515: clamp-basin conditioning must retain its derivative carrier");
    assert_eq!(
        basin_spectrum.conditioning[0],
        BetaSchurSpectralConditioning::Raw
    );
    assert_eq!(
        basin_spectrum.conditioning[1],
        BetaSchurSpectralConditioning::ClampBasin
    );
    assert_abs_diff_eq!(
        reconstructed_basin[[1, 1]],
        -7.997_610e-3 + 2.0e-2,
        epsilon = 1.0e-12,
    );

    let saddle_geometry = ExactAReducedClassification {
        majorizer_metric: basin_geometry.majorizer_metric.clone(),
        clamp_metric: Array2::<f64>::zeros((2, 2)),
    };
    let refusal = factor_dense_reduced_schur_with_exact_a(
        &resolved_negative,
        exact_a,
        Some(&saddle_geometry),
    )
    .expect_err(
        "#2515: exact-A evidence must refuse only after the shared classifier shows \
         that negative curvature remains beyond the clamp basin",
    )
    .to_string();
    assert!(
        ArrowSchurError::rendered_is_indefinite_evidence(&refusal),
        "#2515: the refusal must carry the marker its cross-crate reader matches on, or \
         gam-sae maps this to a fatal defect instead of the same typed \
         `IndefiniteObservedInformation` verdict the dense route returns. Got: {refusal}"
    );

    // A direction genuinely INSIDE the null band, on BOTH sides of zero, is still
    // unit-pinned and must NOT refuse — that is what makes this a two-sided band
    // rather than a positivity requirement. Relative magnitude 1.0e-12, four
    // decades inside the 1.0e-8 floor.
    for (label, band_direction) in [
        ("positive", 4.0e-12_f64),
        ("negative", -4.0e-12_f64),
        ("zero", 0.0_f64),
    ] {
        let in_band = array![[4.0_f64, 0.0], [0.0, band_direction]];
        let null_geometry = ExactAReducedClassification {
            majorizer_metric: basin_geometry.majorizer_metric.clone(),
            clamp_metric: Array2::<f64>::zeros((2, 2)),
        };
        let conditioned =
            factor_dense_reduced_schur_with_exact_a(&in_band, exact_a, Some(&null_geometry))
                .unwrap_or_else(|err| {
                    panic!(
                        "#2515: a {label} direction INSIDE the majorizer-metric null band is a \
                         numerical null and must still be unit-pinned, not refused. Got: {err}"
                    )
                });
        let spectrum = conditioned
            .beta_conditioning
            .as_ref()
            .unwrap_or_else(|| panic!("#2515: the {label} in-band direction must be recorded"));
        let pinned: Vec<usize> = (0..spectrum.conditioning.len())
            .filter(|&index| spectrum.conditioning[index].is_unit_deflated())
            .collect();
        assert_eq!(
            pinned.len(),
            1,
            "#2515: exactly the {label} in-band direction is pinned"
        );
        assert_abs_diff_eq!(spectrum.cond_evals[pinned[0]], 1.0, epsilon = 0.0);
    }
}

/// #2515 — the per-row twin of the reduced-Schur gate above, on the SAME typed
/// majorizer/clamp geometry.
///
/// `factor_spectral_deflated_criterion_row_with_geometry`'s own comment says it deflates
/// "every non-positive/non-finite one", which is the same one-sided predicate and
/// the same defect on the same operator: by Haynsworth the inertia of the exact
/// observed information is the inertia of its per-row blocks plus that of its
/// reduced Schur, so a resolved negative direction can arrive through either and
/// closing only one of them would leave the verdict route-dependent inside the
/// arrow route itself.
#[test]
fn per_row_evidence_classification_prices_a_clamp_basin_before_a_saddle_2515() {
    let d = 2usize;
    let mut block = ArrowRowBlock::new(d, 1);
    block.htt = array![[4.0_f64, 0.0], [0.0, -7.997_610e-3]];
    block.htbeta = array![[1.0_f64], [0.5]];
    block.gt = array![0.0_f64, 0.0];

    let pinned = factor_spectral_deflated_criterion_row_with_geometry(&block, d, false, None)
        .expect("the majorizer policy never refuses on sign")
        .expect("the majorizer policy conditions the negative direction");
    assert_eq!(
        pinned.gauge_deflated_directions, 1,
        "exactly the negative direction is unit-pinned under the majorizer policy"
    );

    let basin_geometry = ExactAClassificationRow {
        delta_tt: array![[0.0_f64, 0.0], [0.0, -4.007_997_610]],
        delta_tbeta: Array2::<f64>::zeros((d, 0)),
        clamp_diag: array![0.0_f64, 2.0e-2],
    };
    let basin = factor_spectral_deflated_criterion_row_with_geometry(
        &block,
        d,
        true,
        Some(&basin_geometry),
    )
    .expect("#2515: clamp-attributable negative row curvature is not a saddle")
    .expect("#2515: the clamp basin produces a conditioned row factor");
    let spectrum = basin
        .deflation_spectrum
        .as_ref()
        .expect("#2515: exact-A row classification carries its raw and priced spectrum");
    assert_abs_diff_eq!(
        spectrum.cond_evals[0],
        -7.997_610e-3 + 2.0e-2,
        epsilon = 1.0e-12,
    );
    assert_eq!(basin.gauge_deflated_directions, 0);

    let saddle_geometry = ExactAClassificationRow {
        delta_tt: basin_geometry.delta_tt.clone(),
        delta_tbeta: Array2::<f64>::zeros((d, 0)),
        clamp_diag: Array1::<f64>::zeros(d),
    };
    let refusal = factor_spectral_deflated_criterion_row_with_geometry(
        &block,
        d,
        true,
        Some(&saddle_geometry),
    )
    .expect_err(
        "#2515: exact-A evidence must refuse a row only after the clamp basin remains negative",
    );
    assert!(
        ArrowSchurError::rendered_is_indefinite_evidence(&refusal),
        "#2515: the per-row refusal must carry the same marker the reduced-Schur one does, \
         or half the verdict is invisible to the cross-crate reader. Got: {refusal}"
    );

    // In-band on both sides: still a unit pin, never a refusal.
    for band_direction in [4.0e-12_f64, -4.0e-12, 0.0] {
        let mut in_band = block.clone();
        in_band.htt = array![[4.0_f64, 0.0], [0.0, band_direction]];
        let null_geometry = ExactAClassificationRow {
            delta_tt: array![[0.0_f64, 0.0], [0.0, band_direction - 4.0]],
            delta_tbeta: Array2::<f64>::zeros((d, 0)),
            clamp_diag: Array1::<f64>::zeros(d),
        };
        let conditioned = factor_spectral_deflated_criterion_row_with_geometry(
            &in_band,
            d,
            true,
            Some(&null_geometry),
        )
        .unwrap_or_else(|err| {
            panic!(
                "#2515: a direction inside the null band ({band_direction:e}) is a \
                 numerical null and must still be unit-pinned, not refused: {err}"
            )
        })
        .unwrap_or_else(|| {
            panic!(
                "#2515: the in-band direction ({band_direction:e}) must still produce a \
                 conditioned factor"
            )
        });
        assert_eq!(
            conditioned.gauge_deflated_directions, 1,
            "#2515: exactly the in-band direction is pinned ({band_direction:e})"
        );
    }
}

/// #2515 — the genuinely matrix-free lane must lift a negative Ritz direction
/// before deciding what its sign means.
///
/// The raw direction below is the measured broken-ladder magnitude.  In the
/// majorizer metric it is resolved, but the bounded clamp restores a positive
/// `1.200239e-2` basin.  Both the SLQ spectral function and the rational
/// ladder's low-rank operator carrier must price that basin; removing the clamp
/// must produce the shared typed saddle refusal.
#[test]
fn matrix_free_exact_a_prices_a_clamp_basin_before_refusing_a_saddle_2515() {
    let exact_a = array![[4.0_f64, 0.0], [0.0, -7.997_610e-3]];
    let majorizer = array![[4.0_f64, 0.0], [0.0, 4.0]];
    let clamp = array![[0.0_f64, 0.0], [0.0, 2.0e-2]];
    let apply_a = |direction: ArrayView1<'_, f64>| exact_a.dot(&direction);
    let metrics = |direction: ArrayView1<'_, f64>| {
        Ok((
            direction.dot(&majorizer.dot(&direction)),
            direction.dot(&clamp.dot(&direction)),
        ))
    };

    let slq = slq_logdet_exact_a_classified(2, apply_a, metrics, 4, 2, 0x2515)
        .expect("#2515: a clamp-attributable negative Ritz direction is a priced basin");
    let expected = 4.0_f64.ln() + (-7.997_610e-3_f64 + 2.0e-2).ln();
    assert_abs_diff_eq!(slq.estimate, expected, epsilon = 1.0e-12);

    let conditioning = exact_a_ritz_conditioning(2, apply_a, metrics, 2, 0x2515)
        .expect("#2515: the rational ladder must receive the same priced Ritz geometry");
    let mut conditioned = exact_a.clone();
    for (direction, &shift) in conditioning
        .directions
        .iter()
        .zip(conditioning.shifts.iter())
    {
        for row in 0..2 {
            for column in 0..2 {
                conditioned[[row, column]] += shift * direction[row] * direction[column];
            }
        }
    }
    assert_abs_diff_eq!(conditioned[[0, 0]], 4.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(conditioned[[0, 1]], 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        conditioned[[1, 1]],
        -7.997_610e-3 + 2.0e-2,
        epsilon = 1.0e-12,
    );

    // Drive both production matrix-free entry arms on an actual arrow system.
    // A_tt=1, A_tbeta=1, A_betabeta=1/2 gives S_A=-1/2.  With
    // B_tt=3 and E_tt=2 the lifted Schur direction has B curvature 3/2 and
    // basin curvature -1/2+2=3/2.
    let exact_a_system = |clamp_value: f64| {
        let mut system = ArrowSchurSystem::new(1, 1, 1);
        system.rows[0].htt[[0, 0]] = 1.0;
        system.rows[0].htbeta[[0, 0]] = 1.0;
        system.hbb[[0, 0]] = 0.5;
        system.exact_a_classification = Some(ExactAClassificationGeometry {
            rows: vec![ExactAClassificationRow {
                delta_tt: array![[-2.0_f64]],
                delta_tbeta: Array2::<f64>::zeros((1, 0)),
                clamp_diag: array![clamp_value],
            }]
            .into(),
            border_indices: Arc::from([] as [usize; 0]),
        });
        system
    };
    let options = ArrowSolveOptions::direct()
        .with_newton_schur_tikhonov(SPECTRAL_DEFLATION_REL_FLOOR)
        .with_indefinite_refusing_evidence_unit_deflation(
            SPECTRAL_DEFLATION_REL_FLOOR,
        );
    let basin_system = exact_a_system(2.0);
    let (row_logdet, slq_schur) = matrix_free_arrow_evidence_log_det_surrogate(
        &basin_system,
        0.0,
        0.0,
        &options,
        4,
        1,
        0x2515,
        None,
    )
    .expect("#2515: the SLQ production arm must price the clamp basin");
    assert_abs_diff_eq!(row_logdet, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(slq_schur, 1.5_f64.ln(), epsilon = 1.0e-12);

    let mut rational_lane = SurrogateLaneState::new(SurrogateLaneConfig {
        num_probes: 4,
        seed: 0x2515,
        rel_tol: 1.0e-10,
        power_iters: 4,
        cg_rel_tol: 1.0e-12,
        cg_max_iters: 64,
        deflation_max_rank: 0,
        deflation_subspace_iters: 1,
        deflation_target_std_err_rel: 1.0,
    });
    let (_, rational_schur) = matrix_free_arrow_evidence_log_det_surrogate(
        &basin_system,
        0.0,
        0.0,
        &options,
        4,
        1,
        0x2515,
        Some(&mut rational_lane),
    )
    .expect("#2515: the rational production arm must solve the priced basin operator");
    assert!(rational_schur.is_finite());
    assert_abs_diff_eq!(rational_schur, 1.5_f64.ln(), epsilon = 1.0e-7);

    let no_clamp = Array2::<f64>::zeros((2, 2));
    let saddle_metrics = |direction: ArrayView1<'_, f64>| {
        Ok((
            direction.dot(&majorizer.dot(&direction)),
            direction.dot(&no_clamp.dot(&direction)),
        ))
    };
    let refusal = slq_logdet_exact_a_classified(2, apply_a, saddle_metrics, 4, 2, 0x2515)
        .expect_err("#2515: negative curvature beyond the clamp basin is a saddle");
    assert!(
        ArrowSchurError::rendered_is_indefinite_evidence(&refusal),
        "#2515: matrix-free and dense saddle refusals must carry one typed marker: {refusal}"
    );

    let mut rational_saddle_lane = SurrogateLaneState::new(SurrogateLaneConfig {
        num_probes: 4,
        seed: 0x2515,
        rel_tol: 1.0e-10,
        power_iters: 4,
        cg_rel_tol: 1.0e-12,
        cg_max_iters: 64,
        deflation_max_rank: 0,
        deflation_subspace_iters: 1,
        deflation_target_std_err_rel: 1.0,
    });
    let rational_refusal = matrix_free_arrow_evidence_log_det_surrogate(
        &exact_a_system(0.0),
        0.0,
        0.0,
        &options,
        4,
        1,
        0x2515,
        Some(&mut rational_saddle_lane),
    )
    .expect_err("#2515: the rational lane must refuse a genuine saddle")
    .to_string();
    assert!(
        ArrowSchurError::rendered_is_indefinite_evidence(&rational_refusal),
        "#2515: rational and dense saddle refusals must carry one typed marker: \
         {rational_refusal}"
    );
}
