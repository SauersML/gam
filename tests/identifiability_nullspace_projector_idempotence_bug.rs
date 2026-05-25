use faer::sparse::{SparseColMat, Triplet};
use gam::basis::apply_sum_to_zero_constraint_sparse;
use ndarray::Array2;

fn dense_from_sparse(s: &SparseColMat<usize, f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((s.nrows(), s.ncols()));
    let (sym, vals) = s.parts();
    let col_ptr = sym.col_ptr();
    let row_idx = sym.row_idx();
    for j in 0..s.ncols() {
        for idx in col_ptr[j]..col_ptr[j + 1] {
            out[[row_idx[idx], j]] = vals[idx];
        }
    }
    out
}

#[test]
fn sparse_sum_to_zero_transform_acts_like_true_projector_under_zz_t() {
    let triplets = vec![
        Triplet::new(0usize, 0usize, 1.0),
        Triplet::new(1, 1, 1.0),
        Triplet::new(2, 2, 1.0),
        Triplet::new(3, 0, 1.0),
        Triplet::new(3, 1, 1.0),
        Triplet::new(3, 2, 1.0),
    ];
    let basis = SparseColMat::<usize, f64>::try_new_from_triplets(4, 3, &triplets)
        .expect("sparse basis from triplets");
    let (_constrained, z) = apply_sum_to_zero_constraint_sparse(&basis, None)
        .expect("sum-to-zero transform should build");

    let p = z.dot(&z.t());
    let p2 = p.dot(&p);
    let err = (&p2 - &p).mapv(|x| x * x).sum().sqrt();
    assert!(
        err <= 1e-12,
        "Expected ZZ^T to be idempotent to machine precision for a null-space projector, but ||P^2 - P||_F = {err:.3e}"
    );

    let b_dense = dense_from_sparse(&basis);
    let projected = b_dense.dot(&p);
    let ctp = projected.t().dot(&ndarray::Array1::ones(projected.nrows()));
    let max_abs = ctp.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    assert!(
        max_abs <= 1e-12,
        "Projected basis should satisfy the sum-to-zero constraint column-wise, but max |1^T B P| = {max_abs:.3e}"
    );
}
