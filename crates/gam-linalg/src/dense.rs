//! Allocation-free kernels for dense `ndarray` matrices.
//!
//! These small operations intentionally preserve scalar iteration order.  Use
//! [`crate::faer_ndarray`] for allocating BLAS-sized products and this module
//! when a caller owns the output buffer or needs an accumulating operation.

use ndarray::{ArrayBase, ArrayView1, ArrayViewMut1, Data, Ix2};

#[inline]
pub fn matvec_into<S: Data<Elem = f64>>(
    matrix: &ArrayBase<S, Ix2>,
    x: ArrayView1<'_, f64>,
    mut out: ArrayViewMut1<'_, f64>,
) {
    assert_eq!(matrix.ncols(), x.len());
    assert_eq!(matrix.nrows(), out.len());
    for (row, out_value) in matrix.rows().into_iter().zip(out.iter_mut()) {
        *out_value = row.dot(&x);
    }
}

#[inline]
pub fn matvec_scaled_add_into<S: Data<Elem = f64>>(
    matrix: &ArrayBase<S, Ix2>,
    x: ArrayView1<'_, f64>,
    scale: f64,
    mut out: ArrayViewMut1<'_, f64>,
) {
    assert_eq!(matrix.ncols(), x.len());
    assert_eq!(matrix.nrows(), out.len());
    if scale == 0.0 {
        return;
    }
    for (row, out_value) in matrix.rows().into_iter().zip(out.iter_mut()) {
        *out_value += scale * row.dot(&x);
    }
}

#[inline]
pub fn transpose_matvec_into<S: Data<Elem = f64>>(
    matrix: &ArrayBase<S, Ix2>,
    x: ArrayView1<'_, f64>,
    mut out: ArrayViewMut1<'_, f64>,
) {
    assert_eq!(matrix.nrows(), x.len());
    assert_eq!(matrix.ncols(), out.len());
    out.fill(0.0);
    transpose_matvec_scaled_add_into(matrix, x, 1.0, out);
}

#[inline]
pub fn transpose_matvec_scaled_add_into<S: Data<Elem = f64>>(
    matrix: &ArrayBase<S, Ix2>,
    x: ArrayView1<'_, f64>,
    scale: f64,
    mut out: ArrayViewMut1<'_, f64>,
) {
    assert_eq!(matrix.nrows(), x.len());
    assert_eq!(matrix.ncols(), out.len());
    if scale == 0.0 {
        return;
    }
    for (row, x_value) in matrix.rows().into_iter().zip(x.iter().copied()) {
        let row_scale = scale * x_value;
        if row_scale == 0.0 {
            continue;
        }
        for (out_value, entry) in out.iter_mut().zip(row.iter().copied()) {
            *out_value += row_scale * entry;
        }
    }
}

#[inline]
pub fn bilinear<S: Data<Elem = f64>>(
    matrix: &ArrayBase<S, Ix2>,
    v: ArrayView1<'_, f64>,
    u: ArrayView1<'_, f64>,
) -> f64 {
    assert_eq!(matrix.ncols(), v.len());
    assert_eq!(matrix.nrows(), u.len());
    matrix
        .rows()
        .into_iter()
        .zip(u.iter().copied())
        .map(|(row, u_value)| u_value * row.dot(&v))
        .sum()
}

/// Compute `trace(left * right)` without materializing the product.
#[inline]
pub fn trace_product<S1: Data<Elem = f64>, S2: Data<Elem = f64>>(
    left: &ArrayBase<S1, Ix2>,
    right: &ArrayBase<S2, Ix2>,
) -> f64 {
    assert_eq!(left.nrows(), left.ncols());
    assert_eq!(left.raw_dim(), right.raw_dim());
    let mut trace = 0.0;
    for (left_row, right_col) in left.rows().into_iter().zip(right.columns().into_iter()) {
        for (left_value, right_value) in left_row.iter().copied().zip(right_col.iter().copied()) {
            trace += left_value * right_value;
        }
    }
    trace
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn matvec_and_scaled_add_use_caller_storage() {
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x = array![1.0, 0.0, -1.0];
        let mut out = Array1::zeros(2);
        matvec_into(&matrix, x.view(), out.view_mut());
        assert_eq!(out, array![-2.0, -2.0]);
        matvec_scaled_add_into(&matrix, x.view(), 2.0, out.view_mut());
        assert_eq!(out, array![-6.0, -6.0]);
        matvec_scaled_add_into(&matrix, x.view(), 0.0, out.view_mut());
        assert_eq!(out, array![-6.0, -6.0]);
    }

    #[test]
    fn transpose_matvec_and_scaled_add_use_caller_storage() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let x = array![1.0, 0.0, 2.0];
        let mut out = array![99.0, 99.0];
        transpose_matvec_into(&matrix, x.view(), out.view_mut());
        assert_eq!(out, array![11.0, 14.0]);
        transpose_matvec_scaled_add_into(&matrix, x.view(), -1.0, out.view_mut());
        assert_eq!(out, array![0.0, 0.0]);
    }

    #[test]
    fn bilinear_matches_identity_dot_product() {
        let matrix = Array2::<f64>::eye(3);
        let u = array![1.0, 2.0, 3.0];
        let v = array![4.0, 5.0, 6.0];
        assert_eq!(bilinear(&matrix, v.view(), u.view()), 32.0);
    }

    #[test]
    fn trace_product_matches_materialized_product() {
        let left = array![[1.0, 2.0], [3.0, 4.0]];
        let right = array![[5.0, 6.0], [7.0, 8.0]];
        assert_eq!(trace_product(&left, &right), left.dot(&right).diag().sum());
    }
}
