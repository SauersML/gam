//! Leaf dense / design-matrix linear-algebra helpers for the unified REML
//! evaluator.
//!
//! These are pure relocation of the matvec, transpose-matvec, bilinear-form,
//! column-extraction, design-apply, and trace-of-product utilities that the
//! outer objective/gradient/Hessian code in [`super`] uses. They operate only
//! on `ndarray` and [`DesignMatrix`] types and hold no module state.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

use gam_linalg::matrix::DesignMatrix;

#[inline]
pub(crate) fn dense_matvec_into(
    matrix: &Array2<f64>,
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
pub(crate) fn dense_matvec_scaled_add_into(
    matrix: &Array2<f64>,
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
pub(crate) fn dense_transpose_matvec_into(
    matrix: &Array2<f64>,
    x: ArrayView1<'_, f64>,
    mut out: ArrayViewMut1<'_, f64>,
) {
    assert_eq!(matrix.nrows(), x.len());
    assert_eq!(matrix.ncols(), out.len());
    out.fill(0.0);
    dense_transpose_matvec_scaled_add_into(matrix, x, 1.0, out);
}

#[inline]
pub(crate) fn dense_transpose_matvec_scaled_add_into(
    matrix: &Array2<f64>,
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
pub(crate) fn dense_bilinear(
    matrix: &Array2<f64>,
    v: ArrayView1<'_, f64>,
    u: ArrayView1<'_, f64>,
) -> f64 {
    assert_eq!(matrix.ncols(), v.len());
    assert_eq!(matrix.nrows(), u.len());
    let mut total = 0.0;
    for (row, u_value) in matrix.rows().into_iter().zip(u.iter().copied()) {
        total += u_value * row.dot(&v);
    }
    total
}

pub(crate) fn design_matrix_apply_view(
    design: &DesignMatrix,
    vector: ArrayView1<'_, f64>,
) -> Array1<f64> {
    let mut output = Array1::<f64>::zeros(design.nrows());
    design_matrix_apply_view_into(design, vector, output.view_mut());
    output
}

pub(crate) fn design_matrix_column_into(
    design: &DesignMatrix,
    col: usize,
    mut output: ArrayViewMut1<'_, f64>,
) {
    assert!(col < design.ncols());
    assert_eq!(design.nrows(), output.len());

    if let Some(dense) = design.as_dense() {
        output.assign(&dense.column(col));
        return;
    }

    if let Some(sparse) = design.as_sparse() {
        let matrix = sparse.as_ref();
        output.fill(0.0);
        let (symbolic, values) = matrix.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for idx in col_ptr[col]..col_ptr[col + 1] {
            output[row_idx[idx]] = values[idx];
        }
        return;
    }

    let mut basis = Array1::<f64>::zeros(design.ncols());
    basis[col] = 1.0;
    output.assign(&design.matrixvectormultiply(&basis));
}

pub(crate) fn design_matrix_apply_view_into(
    design: &DesignMatrix,
    vector: ArrayView1<'_, f64>,
    mut output: ArrayViewMut1<'_, f64>,
) {
    assert_eq!(design.ncols(), vector.len());
    assert_eq!(design.nrows(), output.len());

    if let Some(dense) = design.as_dense() {
        dense_matvec_into(dense, vector, output);
        return;
    }

    if let Some(sparse) = design.as_sparse() {
        let matrix = sparse.as_ref();
        output.fill(0.0);
        let (symbolic, values) = matrix.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for col in 0..matrix.ncols() {
            let x = vector[col];
            if x == 0.0 {
                continue;
            }
            for idx in col_ptr[col]..col_ptr[col + 1] {
                output[row_idx[idx]] += values[idx] * x;
            }
        }
        return;
    }

    output.assign(&design.matrixvectormultiply(&vector.to_owned()));
}

pub(crate) fn design_matrix_transpose_apply_view_into(
    design: &DesignMatrix,
    vector: ArrayView1<'_, f64>,
    mut output: ArrayViewMut1<'_, f64>,
) {
    assert_eq!(design.nrows(), vector.len());
    assert_eq!(design.ncols(), output.len());

    if let Some(dense) = design.as_dense() {
        dense_transpose_matvec_into(dense, vector, output);
        return;
    }

    if let Some(sparse) = design.as_sparse() {
        let matrix = sparse.as_ref();
        let (symbolic, values) = matrix.parts();
        let col_ptr = symbolic.col_ptr();
        let row_idx = symbolic.row_idx();
        for col in 0..matrix.ncols() {
            let mut value = 0.0;
            for idx in col_ptr[col]..col_ptr[col + 1] {
                value += values[idx] * vector[row_idx[idx]];
            }
            output[col] = value;
        }
        return;
    }

    output.assign(&design.transpose_vector_multiply(&vector.to_owned()));
}

#[inline]
pub(crate) fn trace_matrix_product(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
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
