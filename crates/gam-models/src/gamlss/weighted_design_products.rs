//! Weighted Gram and cross-product builders for the GAMLSS location-scale
//! Hessian assembly, plus the small dense-matrix utilities they share.
//!
//! These form the `Xᵀ diag(w) X` / `Xᵀ diag(w) Y` kernels (dense and
//! `DesignMatrix`-backed, the latter chunking row blocks for operator-only
//! designs), a signed magnitude floor, an upper→lower symmetric mirror, and a
//! scaled rank-1 accumulate. All are pure shape-checked wrappers around the
//! `faer_ndarray` fast kernels; the only domain error they raise is a
//! `GamlssError::DimensionMismatch` (or a stable bare-string mismatch for the
//! `DesignMatrix` paths). Extracted verbatim from `gamlss.rs` (issue #780) —
//! no behavior change.

use gam_linalg::faer_ndarray::{fast_xt_diag_x, fast_xt_diag_y};
use gam_linalg::matrix::{DesignMatrix, FiniteSignedWeightsView, LinearOperator};
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut2, s};

use super::GamlssError;
use super::exact_design_row_chunks;

pub(super) fn xt_diag_x_dense(
    design: &Array2<f64>,
    diag: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    if design.nrows() != diag.len() {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "xt_diag_x_dense row mismatch: design has {} rows but diag has {} entries",
                design.nrows(),
                diag.len()
            ),
        }
        .into());
    }
    FiniteSignedWeightsView::try_from_array(diag)
        .map_err(|reason| format!("xt_diag_x_dense: {reason}"))?;
    Ok(fast_xt_diag_x(design, diag))
}

pub(super) fn xt_diag_y_dense(
    left: &Array2<f64>,
    diag: &Array1<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    if left.nrows() != diag.len() {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "xt_diag_y_dense row mismatch: left has {} rows but diag has {} entries",
                left.nrows(),
                diag.len()
            ),
        }
        .into());
    }
    if right.nrows() != diag.len() {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "xt_diag_y_dense row mismatch: right has {} rows but diag has {} entries",
                right.nrows(),
                diag.len()
            ),
        }
        .into());
    }
    FiniteSignedWeightsView::try_from_array(diag)
        .map_err(|reason| format!("xt_diag_y_dense: {reason}"))?;
    Ok(fast_xt_diag_y(left, diag, right))
}

pub(super) fn xt_diag_x_design(
    design: &DesignMatrix,
    diag: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    if design.nrows() != diag.len() {
        return Err(format!(
            "xt_diag_x_design row mismatch: design has {} rows but diag has {} entries",
            design.nrows(),
            diag.len()
        ));
    }
    design.xt_diag_x_signed_op(FiniteSignedWeightsView::try_from_array(diag)?)
}

pub(super) fn xt_diag_y_design(
    left: &DesignMatrix,
    diag: &Array1<f64>,
    right: &DesignMatrix,
) -> Result<Array2<f64>, String> {
    if left.nrows() != diag.len() {
        return Err(format!(
            "xt_diag_y_design row mismatch: left has {} rows but diag has {} entries",
            left.nrows(),
            diag.len()
        ));
    }
    if right.nrows() != diag.len() {
        return Err(format!(
            "xt_diag_y_design row mismatch: right has {} rows but diag has {} entries",
            right.nrows(),
            diag.len()
        ));
    }
    FiniteSignedWeightsView::try_from_array(diag)
        .map_err(|reason| format!("xt_diag_y_design: {reason}"))?;
    if let (Some(left_dense), Some(right_dense)) = (left.as_dense_ref(), right.as_dense_ref()) {
        return xt_diag_y_dense(left_dense, diag, right_dense);
    }

    let mut out = Array2::<f64>::zeros((left.ncols(), right.ncols()));
    for rows in exact_design_row_chunks(diag.len(), left.ncols() + right.ncols()) {
        let left_chunk = left
            .try_row_chunk(rows.clone())
            .map_err(|e| format!("xt_diag_y_design left row chunk materialization failed: {e}"))?;
        let right_chunk = right
            .try_row_chunk(rows.clone())
            .map_err(|e| format!("xt_diag_y_design right row chunk materialization failed: {e}"))?;
        out += &fast_xt_diag_y(&left_chunk, &diag.slice(s![rows]), &right_chunk);
    }
    Ok(out)
}

pub(super) fn mirror_upper_to_lower(target: &mut Array2<f64>) {
    for i in 0..target.nrows() {
        for j in 0..i {
            target[[i, j]] = target[[j, i]];
        }
    }
}

#[inline]
pub(super) fn scaled_outer_add(
    mut target: ArrayViewMut2<'_, f64>,
    scale: f64,
    left: ArrayView1<'_, f64>,
    right: ArrayView1<'_, f64>,
) {
    let n_left = left.len();
    let n_right = right.len();
    for i in 0..n_left {
        // SAFETY: `i < left.len()` by loop construction; target rows match the
        // caller-selected left block.
        let scaled_left = unsafe { *left.uget(i) } * scale;
        for j in 0..n_right {
            // SAFETY: `j < right.len()` by loop construction; target columns
            // match the caller-selected right block.
            unsafe {
                *target.uget_mut((i, j)) += scaled_left * *right.uget(j);
            }
        }
    }
}
