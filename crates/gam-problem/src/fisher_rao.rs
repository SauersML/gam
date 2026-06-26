//! Fisher–Rao precision-weight normalization for response-geometry REML.
//!
//! A caller may supply the per-observation Fisher–Rao precision metric coupling
//! the tangent-coordinate residuals as a 1-D vector (a shared isotropic scale
//! per row), a single 2-D `(dim, dim)` matrix (shared across all rows), or a
//! full 3-D `(n_rows, dim, dim)` stack. This broadcasts any of those to the
//! canonical per-row block form and validates finiteness, per-row symmetry, and
//! positive semidefiniteness. A precision metric induces the squared residual
//! `rᵀ W r`, which must be non-negative for every residual `r`; that holds iff
//! each block is PSD (all eigenvalues `≥ 0`). Symmetry plus a non-negative
//! diagonal is **not** sufficient — e.g. `[[1, 2], [2, 1]]` is symmetric with a
//! non-negative diagonal yet `z = (1, −1)` gives `zᵀ W z = −2 < 0`. The
//! Cholesky-whitening REML consumer needs the stronger positive-definite
//! condition (all eigenvalues `> 0`), exposed via
//! [`normalize_fisher_rao_blocks_pd`]. Single source of truth shared by the
//! `response_geometry_normalize_fisher_rao` FFI shim and any core consumer.

use faer::Side;
use gam_linalg::faer_ndarray::FaerEigh;
use ndarray::{Array2, Array3, ArrayViewD, IxDyn};

/// Required definiteness of each per-row Fisher–Rao precision block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FisherRaoDefiniteness {
    /// Metric / semi-metric use: `rᵀ W r ≥ 0` for all `r`. Rank-deficient
    /// (singular) blocks are accepted; only indefinite blocks are rejected.
    PositiveSemidefinite,
    /// Cholesky-whitening use: the factorization `L Lᵀ = W` needs strict
    /// positive-definiteness, so a zero eigenvalue is rejected too.
    PositiveDefinite,
}

/// Broadcast and validate a Fisher–Rao weight array into `(n_rows, dim, dim)`
/// **positive-semidefinite** precision blocks (the general metric API). Accepts
/// a 1-D `(n_rows,)` isotropic scale, a 2-D `(dim, dim)` shared matrix, or a 3-D
/// `(n_rows, dim, dim)` stack. Rank-deficient (PSD-singular) blocks are
/// accepted; indefinite blocks are rejected. Use
/// [`normalize_fisher_rao_blocks_pd`] for the Cholesky-whitening path.
pub fn normalize_fisher_rao_blocks(
    arr: ArrayViewD<'_, f64>,
    n_rows: usize,
    dim: usize,
) -> Result<Array3<f64>, String> {
    normalize_fisher_rao_blocks_with(
        arr,
        n_rows,
        dim,
        FisherRaoDefiniteness::PositiveSemidefinite,
    )
}

/// Broadcast and validate Fisher–Rao weight blocks requiring each block to be
/// **positive-definite**, as the Cholesky-whitening REML path needs (a singular
/// block has no `L Lᵀ = W` factor). Same broadcasting rules as
/// [`normalize_fisher_rao_blocks`]; the difference is the per-block spectrum
/// must be strictly positive rather than merely non-negative.
pub fn normalize_fisher_rao_blocks_pd(
    arr: ArrayViewD<'_, f64>,
    n_rows: usize,
    dim: usize,
) -> Result<Array3<f64>, String> {
    normalize_fisher_rao_blocks_with(arr, n_rows, dim, FisherRaoDefiniteness::PositiveDefinite)
}

fn normalize_fisher_rao_blocks_with(
    arr: ArrayViewD<'_, f64>,
    n_rows: usize,
    dim: usize,
    definiteness: FisherRaoDefiniteness,
) -> Result<Array3<f64>, String> {
    if !arr.iter().all(|v| v.is_finite()) {
        return Err("fisher_rao_w must contain only finite values".to_string());
    }
    let shape = arr.shape().to_vec();
    let out: Array3<f64> = match arr.ndim() {
        1 => {
            if shape[0] != n_rows {
                return Err(format!(
                    "fisher_rao_w vector must have length {n_rows}; got {}",
                    shape[0]
                ));
            }
            let mut block = Array3::<f64>::zeros((n_rows, dim, dim));
            for row in 0..n_rows {
                let value = arr[IxDyn(&[row])];
                for d in 0..dim {
                    block[[row, d, d]] = value;
                }
            }
            block
        }
        2 => {
            if shape[0] != dim || shape[1] != dim {
                return Err(format!(
                    "fisher_rao_w matrix must have shape ({dim}, {dim}); got ({}, {})",
                    shape[0], shape[1]
                ));
            }
            let mut block = Array3::<f64>::zeros((n_rows, dim, dim));
            for row in 0..n_rows {
                for r in 0..dim {
                    for c in 0..dim {
                        block[[row, r, c]] = arr[IxDyn(&[r, c])];
                    }
                }
            }
            block
        }
        3 => {
            if shape[0] != n_rows || shape[1] != dim || shape[2] != dim {
                return Err(format!(
                    "fisher_rao_w must have shape ({n_rows}, {dim}, {dim}); got ({}, {}, {})",
                    shape[0], shape[1], shape[2]
                ));
            }
            let mut block = Array3::<f64>::zeros((n_rows, dim, dim));
            for row in 0..n_rows {
                for r in 0..dim {
                    for c in 0..dim {
                        block[[row, r, c]] = arr[IxDyn(&[row, r, c])];
                    }
                }
            }
            block
        }
        _ => return Err("fisher_rao_w must be a 1-D, 2-D, or 3-D numeric array".to_string()),
    };
    for row in 0..n_rows {
        for r in 0..dim {
            for c in 0..dim {
                let a = out[[row, r, c]];
                let b = out[[row, c, r]];
                if (a - b).abs() > 1.0e-10 * (1.0 + a.abs() + b.abs()) {
                    return Err("fisher_rao_w must be symmetric in every row block".to_string());
                }
            }
            if out[[row, r, r]] < 0.0 {
                return Err("fisher_rao_w diagonal entries must be non-negative".to_string());
            }
        }
        validate_block_definiteness(out.index_axis(ndarray::Axis(0), row), row, definiteness)?;
    }
    Ok(out)
}

/// Validate that a single symmetric `(dim, dim)` precision block has the
/// required definiteness by checking its eigenvalue spectrum. A precision
/// metric must be PSD so that the induced squared residual `rᵀ W r` is never
/// negative; the Cholesky-whitening path additionally needs PD. The threshold
/// is relative to the block's spectral scale (its largest eigenvalue magnitude)
/// so that the check is invariant to the units of the metric.
fn validate_block_definiteness(
    block: ndarray::ArrayView2<'_, f64>,
    row: usize,
    definiteness: FisherRaoDefiniteness,
) -> Result<(), String> {
    if block.nrows() == 0 {
        return Ok(());
    }
    // Symmetrize before the eigensolve so the spectrum is exactly real; the
    // off-diagonals already match to within the symmetry tolerance checked above.
    let mut symmetric = Array2::<f64>::zeros((block.nrows(), block.ncols()));
    for i in 0..block.nrows() {
        for j in 0..block.ncols() {
            symmetric[[i, j]] = 0.5 * (block[[i, j]] + block[[j, i]]);
        }
    }
    let (eigenvalues, _) = symmetric.eigh(Side::Lower).map_err(|err| {
        format!("fisher_rao_w row {row} eigendecomposition for definiteness check failed: {err}")
    })?;
    let spectral_scale = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()))
        .max(1.0);
    let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    // Relative spectral tolerance: a block is treated as PSD when its smallest
    // eigenvalue is no more negative than this fraction of its spectral scale,
    // absorbing the rounding of the symmetric eigensolve.
    let tol = 1.0e-10 * spectral_scale;
    match definiteness {
        FisherRaoDefiniteness::PositiveSemidefinite => {
            if min_eigenvalue < -tol {
                return Err(format!(
                    "fisher_rao_w row {row} must be positive semidefinite (a precision metric \
                     induces the squared residual rᵀ W r ≥ 0); smallest eigenvalue {min_eigenvalue} \
                     is negative"
                ));
            }
        }
        FisherRaoDefiniteness::PositiveDefinite => {
            if min_eigenvalue <= tol {
                return Err(format!(
                    "fisher_rao_w row {row} must be positive definite for Cholesky whitening; \
                     smallest eigenvalue {min_eigenvalue} is not strictly positive"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn block_2x2(values: [[f64; 2]; 2]) -> Array2<f64> {
        let mut block = Array2::<f64>::zeros((2, 2));
        for r in 0..2 {
            for c in 0..2 {
                block[[r, c]] = values[r][c];
            }
        }
        block
    }

    #[test]
    fn indefinite_block_symmetric_nonneg_diagonal_is_rejected_as_not_psd() {
        // [[1, 2], [2, 1]] is symmetric with a non-negative diagonal but has
        // eigenvalues {3, -1}; z = (1, -1) gives zᵀ W z = -2 < 0, so it is not a
        // valid precision metric. The old symmetry + diagonal checks accepted it.
        let block = block_2x2([[1.0, 2.0], [2.0, 1.0]]);
        let err = normalize_fisher_rao_blocks(block.view().into_dyn(), 4, 2)
            .expect_err("indefinite block must be rejected by the PSD metric API");
        assert!(
            err.contains("positive semidefinite"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn psd_block_is_accepted_by_metric_api_and_broadcast() {
        // [[2, 1], [1, 2]] has eigenvalues {1, 3} > 0 (PSD, in fact PD).
        let block = block_2x2([[2.0, 1.0], [1.0, 2.0]]);
        let n_rows = 3;
        let out = normalize_fisher_rao_blocks(block.view().into_dyn(), n_rows, 2)
            .expect("a genuinely PSD block must be accepted");
        assert_eq!(out.shape(), &[n_rows, 2, 2]);
        for row in 0..n_rows {
            assert_eq!(out[[row, 0, 0]], 2.0);
            assert_eq!(out[[row, 1, 0]], 1.0);
            assert_eq!(out[[row, 0, 1]], 1.0);
            assert_eq!(out[[row, 1, 1]], 2.0);
        }
    }

    #[test]
    fn pd_block_passes_the_cholesky_path() {
        // Eigenvalues {1, 3}, both strictly positive: valid for Cholesky whitening.
        let block = block_2x2([[2.0, 1.0], [1.0, 2.0]]);
        normalize_fisher_rao_blocks_pd(block.view().into_dyn(), 2, 2)
            .expect("a positive-definite block must pass the Cholesky (PD) path");
    }

    #[test]
    fn psd_singular_block_passes_metric_api_but_is_rejected_on_cholesky_path() {
        // [[1, 1], [1, 1]] has eigenvalues {0, 2}: PSD but singular. The metric
        // API must accept it (rᵀ W r ≥ 0), while the Cholesky-whitening path
        // requires strict positive-definiteness and must reject it.
        let block = block_2x2([[1.0, 1.0], [1.0, 1.0]]);
        normalize_fisher_rao_blocks(block.view().into_dyn(), 2, 2)
            .expect("a PSD-singular block must be accepted by the metric API");
        let err = normalize_fisher_rao_blocks_pd(block.view().into_dyn(), 2, 2)
            .expect_err("a singular block has no Cholesky factor and must be rejected");
        assert!(
            err.contains("positive definite"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn isotropic_scale_vector_remains_accepted() {
        // A 1-D isotropic scale becomes diag(value) per row: PSD when non-negative.
        let scales = ndarray::Array1::from(vec![0.5_f64, 2.0, 1.0]);
        let out = normalize_fisher_rao_blocks(scales.view().into_dyn(), 3, 2)
            .expect("non-negative isotropic scales are PSD");
        assert_eq!(out[[1, 0, 0]], 2.0);
        assert_eq!(out[[1, 1, 1]], 2.0);
        assert_eq!(out[[1, 0, 1]], 0.0);
    }

    #[test]
    fn per_row_indefinite_block_is_rejected_with_its_row_index() {
        // A 3-D stack whose second row block is indefinite must be rejected and
        // name that row, confirming the check runs per row block.
        let mut stack = ndarray::Array3::<f64>::zeros((2, 2, 2));
        for row in 0..2 {
            stack[[row, 0, 0]] = 2.0;
            stack[[row, 1, 1]] = 2.0;
        }
        stack[[1, 0, 1]] = 3.0;
        stack[[1, 1, 0]] = 3.0; // eigenvalues {-1, 5}: indefinite.
        let err = normalize_fisher_rao_blocks(stack.view().into_dyn(), 2, 2)
            .expect_err("the indefinite row block must be rejected");
        assert!(err.contains("row 1"), "unexpected error message: {err}");
    }

    #[test]
    fn non_square_dynamic_input_is_still_rejected_by_shape_check() {
        // Sanity: the eigenvalue addition must not regress the existing shape
        // validation for a malformed 2-D matrix.
        let block = Array2::<f64>::zeros((3, 2));
        let err = normalize_fisher_rao_blocks(block.view().into_dyn(), 4, 2)
            .expect_err("a (3, 2) matrix is not a valid (2, 2) shared block");
        assert!(err.contains("shape"), "unexpected error message: {err}");
    }
}
