use faer::Side;
use gam::faer_ndarray::FaerCholesky;
use gam::linalg::sparse_exact::{
    TakahashiInverse, dense_to_sparse, dense_to_sparse_symmetric_upper, factorize_simplicial,
    factorize_sparse_spd, logdet_from_factor,
};
use ndarray::{Array2, array};

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
