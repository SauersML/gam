use crate::estimate::EstimationError;
use crate::matrix::SymmetricMatrix;
use faer::sparse::SparseColMat;
use faer::sparse::Triplet;
use ndarray::Array1;

/// Add `lambda * d2[col]` to the diagonal of a sparse upper-triangular CSC matrix.
///
/// Fast path: if diagonal entries already exist in the symbolic structure,
/// the value buffer is cloned and mutated without symbolic reconstruction.
/// Slow path: triplet rebuild that inserts any missing diagonal entries.
pub(super) fn add_scaled_diagonal_to_upper_sparse(
    matrix: &SparseColMat<usize, f64>,
    lambda: f64,
    d2: &[f64],
) -> Result<SparseColMat<usize, f64>, EstimationError> {
    let (symbolic, values) = matrix.parts();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();

    // Fast path: if diagonal entries already exist in the sparse structure,
    // clone values and modify in-place (avoids triplet reconstruction).
    let has_all_diags = (0..matrix.ncols()).all(|col| {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        row_idx[start..end].contains(&col)
    });

    if has_all_diags {
        let mut new_values = values.to_vec();
        for col in 0..matrix.ncols() {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                if row_idx[idx] == col {
                    new_values[idx] += lambda * d2[col];
                    break;
                }
            }
        }
        // Rebuild from triplets using the known structure.
        // This is much faster than the general slow path below because we
        // know exactly which entries exist and their positions.
        let mut triplets = Vec::with_capacity(values.len());
        for col in 0..matrix.ncols() {
            for idx in col_ptr[col]..col_ptr[col + 1] {
                triplets.push(Triplet::new(row_idx[idx], col, new_values[idx]));
            }
        }
        return SparseColMat::try_new_from_triplets(matrix.nrows(), matrix.ncols(), &triplets)
            .map_err(|_| {
                EstimationError::InvalidInput(
                    "failed to rebuild sparse matrix with diagonal update".to_string(),
                )
            });
    }

    // Slow path: diagonal entries missing from structure, must rebuild.
    let mut triplets = Vec::with_capacity(values.len() + matrix.ncols());
    for col in 0..matrix.ncols() {
        let mut saw_diag = false;
        for idx in col_ptr[col]..col_ptr[col + 1] {
            let row = row_idx[idx];
            let mut value = values[idx];
            if row == col {
                value += lambda * d2[col];
                saw_diag = true;
            }
            triplets.push(Triplet::new(row, col, value));
        }
        if !saw_diag {
            triplets.push(Triplet::new(col, col, lambda * d2[col]));
        }
    }
    SparseColMat::try_new_from_triplets(matrix.nrows(), matrix.ncols(), &triplets).map_err(|_| {
        EstimationError::InvalidInput("failed to add diagonal to sparse matrix".to_string())
    })
}

/// Add `delta * d2[col]` to every diagonal entry of an already-built sparse
/// CSC matrix, mutating its value buffer in place. The symbolic structure is
/// reused — no reallocation occurs. Errors if any diagonal entry is missing
/// from the sparsity pattern (which would indicate a real bug, not a fallback
/// case; callers must materialize diagonals before the first call).
pub(super) fn update_scaled_diagonal_in_place(
    m: &mut SparseColMat<usize, f64>,
    delta: f64,
    d2: &[f64],
) -> Result<(), String> {
    if delta == 0.0 {
        return Ok(());
    }
    let ncols = m.ncols();
    let (symbolic, values) = m.parts_mut();
    let col_ptr = symbolic.col_ptr();
    let row_idx = symbolic.row_idx();
    for col in 0..ncols {
        let start = col_ptr[col];
        let end = col_ptr[col + 1];
        let mut found = false;
        for idx in start..end {
            if row_idx[idx] == col {
                values[idx] += delta * d2[col];
                found = true;
                break;
            }
        }
        if !found {
            return Err(format!(
                "update_scaled_diagonal_in_place: diagonal entry missing for column {col}"
            ));
        }
    }
    Ok(())
}

/// Compute the per-coordinate LM damping scale D²[i] from the penalized
/// Hessian diagonal. Uses `max(H_diag[i], ε)` clamped to `[D2_MIN, D2_MAX]`
/// so that D² stays in a numerically safe range and is strictly positive.
///
/// Using the full penalized-Hessian diagonal (X'WX + Sρ)_ii means D² reflects
/// the actual curvature in each coordinate and is invariant to the choice of
/// whether curvature comes from Fisher or observed information — we always
/// clamp to the Fisher-curvature floor via `max(·, ε)`.
pub(super) fn compute_lm_d2(h: &SymmetricMatrix) -> Array1<f64> {
    pub(crate) const D2_EPS: f64 = 1e-8;
    pub(crate) const D2_MIN: f64 = 1e-8;
    pub(crate) const D2_MAX: f64 = 1e8;
    let p = h.nrows();
    let mut d2 = Array1::<f64>::zeros(p);
    match h {
        SymmetricMatrix::Dense(mat) => {
            for i in 0..p {
                d2[i] = mat[[i, i]].max(D2_EPS).clamp(D2_MIN, D2_MAX);
            }
        }
        SymmetricMatrix::Sparse(mat) => {
            let (symbolic, values) = mat.parts();
            let col_ptr = symbolic.col_ptr();
            let row_idx = symbolic.row_idx();
            for col in 0..p {
                let start = col_ptr[col];
                let end = col_ptr[col + 1];
                let mut diag_val = 0.0_f64;
                for idx in start..end {
                    if row_idx[idx] == col {
                        diag_val = values[idx];
                        break;
                    }
                }
                d2[col] = diag_val.max(D2_EPS).clamp(D2_MIN, D2_MAX);
            }
        }
    }
    d2
}
