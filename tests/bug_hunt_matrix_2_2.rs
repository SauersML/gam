use faer::{Mat, Side};
use gam::linalg::faer_ndarray::factorize_symmetricwith_fallback;
use gam::linalg::low_rank_weight::LowRankWeight;
use gam::linalg::matrix::{
    ConditionedDesign, DenseDesignMatrix, DesignMatrix, LinearOperator, PsdWeightsView,
    SignedWeightsView, xt_diag_x_psd, xt_diag_x_signed, xt_diag_x_symmetric,
};
use ndarray::{Array2, array};

#[test]
fn xt_diag_x_symmetric_matches_dense_reference_for_spd_weights() {
    let x = array![
        [1.0, 2.0, -1.0],
        [0.0, -3.0, 2.0],
        [4.0, 1.0, 0.5],
        [2.0, -2.0, 3.0]
    ];
    let w = array![0.2, 1.5, 0.7, 2.1];
    let design = DesignMatrix::Dense(DenseDesignMatrix::from(x.clone()));
    let got = xt_diag_x_symmetric(&design, &w)
        .expect("xt_diag_x_symmetric should assemble X^T W X for SPD weights")
        .to_dense();
    let wx = Array2::from_shape_fn((x.nrows(), x.ncols()), |(i, j)| w[i] * x[[i, j]]);
    let expected = x.t().dot(&wx);

    let mut max_sym_err: f64 = 0.0;
    let mut max_ref_err: f64 = 0.0;
    for i in 0..got.nrows() {
        for j in 0..got.ncols() {
            max_sym_err = max_sym_err.max((got[[i, j]] - got[[j, i]]).abs());
            max_ref_err = max_ref_err.max((got[[i, j]] - expected[[i, j]]).abs());
        }
    }
    assert!(
        max_sym_err <= 1e-12 && max_ref_err <= 1e-9,
        "xt_diag_x_symmetric should be symmetric to machine precision and match dense reference within 1e-9"
    );
}

#[test]
fn xt_diag_x_signed_and_psd_paths_are_consistent_with_weight_semantics() {
    let x = array![[1.0, 2.0], [3.0, -1.0], [-2.0, 4.0]];
    let design = DesignMatrix::Dense(DenseDesignMatrix::from(x.clone()));
    let w_signed = array![1.0, -0.5, 2.0];
    let w_psd = array![1.0, 0.5, 2.0];

    let signed = xt_diag_x_signed(&design, SignedWeightsView::from_array(&w_signed))
        .expect("xt_diag_x_signed should accept negative weights")
        .to_dense();
    let signed_ref = x
        .t()
        .dot(&Array2::from_shape_fn((x.nrows(), x.ncols()), |(i, j)| {
            w_signed[i] * x[[i, j]]
        }));
    let psd = xt_diag_x_psd(
        &design,
        PsdWeightsView::try_new(w_psd.view()).expect("nonneg by construction"),
    )
    .expect("xt_diag_x_psd should accept nonnegative weights")
    .to_dense();
    let psd_via_signed = xt_diag_x_signed(&design, SignedWeightsView::from_array(&w_psd))
        .expect("xt_diag_x_signed should match psd path when all weights are nonnegative")
        .to_dense();

    let signed_err = (&signed - &signed_ref)
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    let psd_err = (&psd - &psd_via_signed)
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        signed_err <= 1e-9 && psd_err <= 1e-12,
        "signed variant must preserve negative-weight contributions and psd variant must agree on nonnegative weights"
    );
}

#[test]
fn factorize_symmetric_with_fallback_returns_working_solve_after_cholesky_failure() {
    let a = Mat::from_fn(2, 2, |i, j| if i == j { 0.0 } else { 1.0 });
    let rhs = Mat::from_fn(2, 1, |i, _| if i == 0 { 2.0 } else { -3.0 });
    let factor = factorize_symmetricwith_fallback(a.as_ref(), Side::Lower)
        .expect("fallback factorization should succeed on indefinite symmetric input");
    let x = factor.solve(rhs.as_ref());
    let ax = &a * &x;
    let r0 = ax[(0, 0)] - rhs[(0, 0)];
    let r1 = ax[(1, 0)] - rhs[(1, 0)];
    let rnorm = (r0.abs() + r1.abs()).max(1e-16);
    let bnorm = rhs[(0, 0)].abs() + rhs[(1, 0)].abs();
    let berr = rnorm / bnorm.max(1e-16);
    assert!(
        berr <= 1e-8,
        "factorize_symmetric_with_fallback should produce a solve with backward error at most 1e-8"
    );
}

#[test]
fn conditioned_design_operator_matches_explicit_column_conditioning_for_matvec() {
    let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, -2.0]];
    let design = DesignMatrix::Dense(DenseDesignMatrix::from(x.clone()));
    let conditioned = ConditionedDesign::new(design, vec![(1, 1.0, 2.0)]);
    let v = array![2.0, -3.0];
    let got = conditioned.apply(&v);

    let mut xc = x.clone();
    xc.column_mut(1).mapv_inplace(|z| (z - 1.0) / 2.0);
    let expected = xc.dot(&v);
    let err = (&got - &expected)
        .iter()
        .fold(0.0_f64, |m, z| m.max(z.abs()));
    assert!(
        err <= 1e-12,
        "ConditionedDesignOperator matvec should match explicit lazy column rescaling"
    );
}

#[test]
fn low_rank_weight_assembly_satisfies_d_plus_uu_t_identity() {
    let d = array![1.0, 2.0, 0.5, 3.0];
    let u = array![[1.0, 0.0], [0.5, 1.0], [-1.0, 2.0], [0.0, -0.5]];
    let v = array![0.2, -1.5, 2.0, 0.3];
    let w = LowRankWeight::symmetric(d.view(), u.view())
        .expect("valid low-rank symmetric weight should construct");
    let got = w.apply(v.view());
    let expected = &(&d * &v) + &u.dot(&u.t().dot(&v));
    let err = (&got - &expected)
        .iter()
        .fold(0.0_f64, |m, z| m.max(z.abs()));
    assert!(
        err <= 1e-12,
        "low-rank weight assembly should satisfy (D + U U^T)v = Dv + U(U^T v)"
    );
}

#[test]
fn matrix_error_conditions_surface_through_public_apis() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let design = DesignMatrix::Dense(DenseDesignMatrix::from(x));
    let bad_w = array![1.0];
    let err = xt_diag_x_symmetric(&design, &bad_w).expect_err("row mismatch must be rejected");
    assert!(
        err.contains("row mismatch"),
        "dimension mismatch should surface as an explicit row-mismatch error message"
    );
}
