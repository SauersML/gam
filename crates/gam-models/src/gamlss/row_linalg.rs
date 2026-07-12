//! Small pure leaf linear-algebra helpers shared across the GAMLSS family
//! implementations.

use super::GamlssError;
use ndarray::{Array1, Array2};

pub(crate) fn scale_matrix_rows(
    mat: &Array2<f64>,
    coeffs: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    if mat.nrows() != coeffs.len() {
        return Err(GamlssError::DimensionMismatch {
            reason: format!(
                "row scaling dimension mismatch: matrix has {} rows but coeffs have {} entries",
                mat.nrows(),
                coeffs.len()
            ),
        }
        .into());
    }
    Ok(Array2::from_shape_fn(mat.dim(), |(i, j)| {
        mat[[i, j]] * coeffs[i]
    }))
}
