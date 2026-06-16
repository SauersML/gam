use faer::Side;
use gam::faer_ndarray::FaerCholesky;
use gam::linalg::sparse_exact::{
    TakahashiInverse, build_sparse_penalty_blocks_from_canonical, dense_to_sparse,
    dense_to_sparse_symmetric_upper, factorize_simplicial, factorize_sparse_spd,
    logdet_from_factor,
};
use gam::terms::construction::CanonicalPenalty;
use ndarray::{Array1, Array2, array};

fn make_spd_from_banded(seed_shift: f64, n: usize) -> Array2<f64> {
    let mut b = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        b[[i, i]] = 1.5 + (i as f64) * 0.3 + seed_shift;
        if i + 1 < n {
            b[[i, i + 1]] = 0.2 + (i as f64) * 0.01;
        }
        if i + 2 < n {
            b[[i, i + 2]] = -0.07 + (i as f64) * 0.005;
        }
    }
    let bt = b.t().to_owned();
    let mut a = bt.dot(&b);
    for i in 0..n {
        a[[i, i]] += 0.5;
    }
    a
}

fn dense_inverse_spd(a: &Array2<f64>) -> Array2<f64> {
    let chol = a
        .cholesky(Side::Lower)
        .expect("dense SPD inverse reference requires Cholesky");
    let n = a.nrows();
    let mut identity = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        identity[[i, i]] = 1.0;
    }
    chol.solve_mat(&identity)
}

#[test]
fn sparse_cholesky_reconstructs_random_spd_within_1e9() {
    let a = make_spd_from_banded(0.4, 8);
    let a_sparse =
        dense_to_sparse_symmetric_upper(&a, 0.0).expect("upper sparse conversion should succeed");
    let factor = factorize_sparse_spd(&a_sparse)
        .expect("sparse Cholesky factorization should succeed for SPD matrix");
    let reconstructed = gam::linalg::sparse_exact::assemble_sparse_factor_h_dense(&factor)
        .expect("reconstructing dense matrix from sparse Cholesky factor should succeed");

    let max_abs_diff = a
        .iter()
        .zip(reconstructed.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_abs_diff <= 1e-9,
        "Sparse Cholesky should satisfy L*L^T ≈ A within 1e-9, but max |A - LL^T| was {max_abs_diff:.3e}"
    );
}

#[test]
fn sparse_cholesky_logdet_matches_dense_two_sum_log_diag_l() {
    let a = make_spd_from_banded(0.9, 7);
    let a_sparse =
        dense_to_sparse_symmetric_upper(&a, 0.0).expect("upper sparse conversion should succeed");
    let factor = factorize_sparse_spd(&a_sparse)
        .expect("sparse Cholesky factorization should succeed for SPD matrix");
    let sparse_logdet =
        logdet_from_factor(&factor).expect("logdet from sparse factor should be available");

    let chol = a
        .cholesky(Side::Lower)
        .expect("dense Cholesky should succeed for SPD reference");
    let dense_logdet = 2.0 * chol.diag().iter().map(|d| d.ln()).sum::<f64>();

    assert!(
        (sparse_logdet - dense_logdet).abs() <= 1e-9,
        "logdet from sparse Cholesky should equal 2*sum(log(L_ii)) within 1e-9, but diff was {:.3e}",
        (sparse_logdet - dense_logdet).abs()
    );
}

#[test]
fn takahashi_inverse_diagonal_matches_dense_inverse_diagonal_random_spd() {
    let a = make_spd_from_banded(1.1, 9);
    let a_sparse =
        dense_to_sparse_symmetric_upper(&a, 0.0).expect("upper sparse conversion should succeed");
    let simplicial = factorize_simplicial(&a_sparse)
        .expect("simplicial sparse factorization should succeed for SPD matrix");
    let taka = TakahashiInverse::compute(&simplicial)
        .expect("Takahashi selected inverse should compute for SPD matrix");

    let dense_inv = dense_inverse_spd(&a);
    let dense_diag = dense_inv.diag().to_owned();
    let taka_diag = taka.diagonal();

    let max_abs_diff = dense_diag
        .iter()
        .zip(taka_diag.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_abs_diff <= 1e-9,
        "Takahashi inverse diagonal should match dense A^(-1) diagonal within 1e-9, but max diff was {max_abs_diff:.3e}"
    );
}

#[test]
fn takahashi_trace_hinv_sk_matches_dense_trace_small_problem() {
    let h = make_spd_from_banded(0.2, 6);
    let s_k = array![
        [2.0, 0.3, 0.0, 0.0, 0.0, 0.0],
        [0.3, 1.4, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.2, 1.1, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 1.8, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.4, 1.2, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.9],
    ];
    let h_sparse =
        dense_to_sparse_symmetric_upper(&h, 0.0).expect("upper sparse conversion should succeed");
    let s_sparse = dense_to_sparse(&s_k, 0.0).expect("dense sparse conversion should succeed");

    let simplicial =
        factorize_simplicial(&h_sparse).expect("simplicial factorization should succeed");
    let taka =
        TakahashiInverse::compute(&simplicial).expect("Takahashi selected inverse should compute");
    let sparse_trace = taka.trace_product_sparse(&s_sparse);

    let dense_inv = dense_inverse_spd(&h);
    let dense_trace = (dense_inv.dot(&s_k)).diag().sum();

    assert!(
        (sparse_trace - dense_trace).abs() <= 1e-9,
        "trace(H^(-1) S_k) from Takahashi should match dense reference within 1e-9, but diff was {:.3e}",
        (sparse_trace - dense_trace).abs()
    );
}

#[test]
fn canonical_penalty_blocks_return_some_for_nonoverlap_and_none_for_overlap() {
    let p = 8;
    let non_overlap = vec![
        CanonicalPenalty::from_dense_root_with_mean(array![[1.0, 0.0]], p, Array1::zeros(p)),
        CanonicalPenalty::from_dense_root_with_mean(array![[0.0, 0.0, 2.0]], p, Array1::zeros(p)),
    ];

    let mut non_overlap_adjusted = non_overlap.clone();
    non_overlap_adjusted[0].col_range = 0..2;
    non_overlap_adjusted[0].local = array![[1.0, 0.0], [0.0, 0.5]];
    non_overlap_adjusted[0].positive_eigenvalues = vec![1.0, 0.5];
    non_overlap_adjusted[1].col_range = 4..7;
    non_overlap_adjusted[1].local = array![[2.0, 0.0, 0.0], [0.0, 1.2, 0.1], [0.0, 0.1, 0.9]];
    non_overlap_adjusted[1].positive_eigenvalues = vec![2.0, 1.2, 0.9];

    let maybe_blocks = build_sparse_penalty_blocks_from_canonical(&non_overlap_adjusted, p).expect(
        "building sparse penalty blocks for non-overlapping canonical penalties should succeed",
    );
    let blocks =
        maybe_blocks.expect("non-overlapping canonical penalties should return Some(blocks)");

    assert_eq!(blocks[0].p_start, 0, "first block should start at column 0");
    assert_eq!(blocks[0].p_end, 2, "first block should end at column 2");
    assert_eq!(
        blocks[1].p_start, 4,
        "second block should start at column 4"
    );
    assert_eq!(blocks[1].p_end, 7, "second block should end at column 7");

    let mut overlap = non_overlap_adjusted.clone();
    overlap[1].col_range = 1..4;
    let overlap_result = build_sparse_penalty_blocks_from_canonical(&overlap, p)
        .expect("overlap detection should complete without a low-level construction error");
    assert!(
        overlap_result.is_none(),
        "overlapping canonical penalty supports should return None instead of Some(blocks)"
    );
}

#[test]
fn sparse_penalty_block_invariants_hold_for_constructed_blocks() {
    let p = 7;
    let mut cp =
        CanonicalPenalty::from_dense_root_with_mean(array![[1.0, 0.0, 0.0]], p, Array1::zeros(p));
    cp.col_range = 2..5;
    cp.local = array![[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 2.0]];
    cp.positive_eigenvalues = vec![1.0, 1.2, 2.3];

    let blocks = build_sparse_penalty_blocks_from_canonical(&[cp], p)
        .expect("single canonical penalty block should build without error")
        .expect("single non-overlapping canonical penalty should return Some(blocks)");
    let block = &blocks[0];

    assert!(
        block.p_start <= block.p_end,
        "SparsePenaltyBlock must satisfy p_start <= p_end"
    );
    assert!(
        block.positive_eigenvalues.iter().all(|&ev| ev > 0.0),
        "SparsePenaltyBlock positive_eigenvalues should all be strictly positive"
    );

    let mut all_inside_block = true;
    let (symbolic, values) = block.s_k_sparse.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..block.s_k_sparse.ncols() {
        for idx in col_ptr[col]..col_ptr[col + 1] {
            if values[idx].abs() > 0.0 {
                let row = row_idx[idx];
                if !(block.p_start <= row
                    && row < block.p_end
                    && block.p_start <= col
                    && col < block.p_end)
                {
                    all_inside_block = false;
                }
            }
        }
    }

    assert_eq!(
        block.block_support_strict, all_inside_block,
        "SparsePenaltyBlock block_support_strict should be true iff all non-zero entries lie within [p_start, p_end)"
    );
}
