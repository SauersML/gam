use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerSymmetricFactor, array2_to_matmut};
use crate::linalg::utils::{StableSolver, array_is_finite};
use crate::matrix::SymmetricMatrix;
use crate::types::Coefficients;
use ndarray::{Array1, Array2};

use super::{PirlsPenalty, PirlsWorkspace};

/// Result of the stable penalized least squares solve
#[derive(Clone)]
pub struct StablePLSResult {
    /// Solution vector beta
    pub beta: Coefficients,
    /// Final penalized Hessian matrix (sparse or dense depending on solve path)
    pub penalized_hessian: SymmetricMatrix,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Residual standard deviation estimate.
    ///
    /// Contract: for Gaussian identity models this is the residual standard
    /// deviation (sigma), not the residual variance/dispersion.
    pub standard_deviation: f64,
    /// Ridge added to ensure the SPD solve is well-posed.
    pub ridge_used: f64,
}

/// EDF from an already-factorized dense regularized Hessian (dense path).
///
/// Mirrors `calculate_edfwithworkspace_with_penalty` but accepts the
/// `FaerSymmetricFactor` that PLS already produced, eliminating the redundant
/// second O(p³) factorization inside every PIRLS outer iteration.
pub(super) fn calculate_edfwithworkspace_from_factor(
    factor: &FaerSymmetricFactor,
    penalty: &PirlsPenalty,
    workspace: &mut PirlsWorkspace,
) -> Result<f64, EstimationError> {
    match penalty {
        PirlsPenalty::Dense { e_transformed, .. } => {
            let p = factor.n();
            let r = e_transformed.nrows();
            let mp = ((p - r) as f64).max(0.0);
            if r == 0 {
                return Ok(p as f64);
            }
            if workspace.final_aug_matrix.nrows() != p || workspace.final_aug_matrix.ncols() != r {
                workspace.final_aug_matrix = Array2::zeros((p, r));
            }
            for j in 0..r {
                for i in 0..p {
                    workspace.final_aug_matrix[[i, j]] = e_transformed[[j, i]];
                }
            }
            {
                let mut rhsview = array2_to_matmut(&mut workspace.final_aug_matrix);
                factor.solve_in_place(rhsview.as_mut());
            }
            if workspace.final_aug_matrix.nrows() == p
                && workspace.final_aug_matrix.ncols() == r
                && array_is_finite(&workspace.final_aug_matrix)
            {
                return Ok(edf_from_solution(p, r, mp, e_transformed, |i, j| {
                    workspace.final_aug_matrix[(i, j)]
                }));
            }
            Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })
        }
        PirlsPenalty::Diagonal {
            diag,
            positive_indices,
            ..
        } => {
            let p = factor.n();
            let r = positive_indices.len();
            let mp = ((p - r) as f64).max(0.0);
            if r == 0 {
                return Ok(p as f64);
            }
            if workspace.final_aug_matrix.nrows() != p || workspace.final_aug_matrix.ncols() != r {
                workspace.final_aug_matrix = Array2::zeros((p, r));
            } else {
                workspace.final_aug_matrix.fill(0.0);
            }
            for (col, &idx) in positive_indices.iter().enumerate() {
                workspace.final_aug_matrix[[idx, col]] = 1.0;
            }
            {
                let mut rhsview = array2_to_matmut(&mut workspace.final_aug_matrix);
                factor.solve_in_place(rhsview.as_mut());
            }
            let mut tr = 0.0;
            for (col, &idx) in positive_indices.iter().enumerate() {
                tr += diag[idx] * workspace.final_aug_matrix[[idx, col]];
            }
            Ok((p as f64 - tr).clamp(mp, p as f64))
        }
    }
}

/// EDF from an already-factorized sparse penalized Hessian (sparse path).
///
/// Mirrors `calculate_edf_with_penalty` but accepts the `SparseExactFactor`
/// that PLS already produced, eliminating the redundant second sparse
/// factorization inside every PIRLS outer iteration.
///
/// Only the `PirlsPenalty::Dense` variant is handled because the sparse-native
/// path requires `PirlsPenalty::Dense` (enforced by the caller).
pub(super) fn calculate_edf_from_sparse_factor(
    factor: &crate::linalg::sparse_exact::SparseExactFactor,
    penalty: &PirlsPenalty,
) -> Result<f64, EstimationError> {
    let PirlsPenalty::Dense { e_transformed, .. } = penalty else {
        crate::bail_invalid_estim!("calculate_edf_from_sparse_factor requires PirlsPenalty::Dense");
    };
    // e_transformed has shape (r, p) — cols give the coefficient dimension p.
    let p = e_transformed.ncols();
    let r = e_transformed.nrows();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    let rhs_arr = e_transformed.t().to_owned();
    let sol =
        crate::linalg::sparse_exact::solve_sparse_spdmulti(factor, &rhs_arr).map_err(|_| {
            EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            }
        })?;
    if sol.nrows() == p && sol.ncols() == r && sol.iter().all(|v| v.is_finite()) {
        return Ok(edf_from_solution(p, r, mp, e_transformed, |i, j| {
            sol[[i, j]]
        }));
    }
    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

pub(super) fn calculate_edf(
    penalized_hessian: &SymmetricMatrix,
    e_transformed: &Array2<f64>,
) -> Result<f64, EstimationError> {
    let p = penalized_hessian.ncols();
    let r = e_transformed.nrows();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    let rhs_arr = e_transformed.t().to_owned();
    // Use SymmetricMatrix::factorize() which dispatches to sparse Cholesky
    // for sparse Hessians and dense Cholesky for dense ones.
    let factor =
        penalized_hessian
            .factorize()
            .map_err(|_| EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
    let sol = factor
        .solvemulti(&rhs_arr)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    if sol.nrows() == p && sol.ncols() == r && sol.iter().all(|v| v.is_finite()) {
        return Ok(edf_from_solution(p, r, mp, e_transformed, |i, j| {
            sol[[i, j]]
        }));
    }

    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

pub(super) fn calculate_edf_with_penalty(
    penalized_hessian: &SymmetricMatrix,
    penalty: &PirlsPenalty,
) -> Result<f64, EstimationError> {
    match penalty {
        PirlsPenalty::Dense { e_transformed, .. } => {
            calculate_edf(penalized_hessian, e_transformed)
        }
        PirlsPenalty::Diagonal {
            diag,
            positive_indices,
            ..
        } => calculate_edf_from_diagonal_penalty(penalized_hessian, diag, positive_indices),
    }
}

pub(super) fn calculate_edfwithworkspace(
    penalized_hessian: &Array2<f64>,
    e_transformed: &Array2<f64>,
    workspace: &mut PirlsWorkspace,
) -> Result<f64, EstimationError> {
    let p = penalized_hessian.ncols();
    let r = e_transformed.nrows();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    if workspace.final_aug_matrix.nrows() != p || workspace.final_aug_matrix.ncols() != r {
        workspace.final_aug_matrix = Array2::zeros((p, r));
    }
    for j in 0..r {
        for i in 0..p {
            workspace.final_aug_matrix[[i, j]] = e_transformed[[j, i]];
        }
    }

    let factor = StableSolver::new("pirls edf workspace")
        .factorize(penalized_hessian)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    {
        let mut rhsview = array2_to_matmut(&mut workspace.final_aug_matrix);
        factor.solve_in_place(rhsview.as_mut());
    }
    if workspace.final_aug_matrix.nrows() == p
        && workspace.final_aug_matrix.ncols() == r
        && array_is_finite(&workspace.final_aug_matrix)
    {
        return Ok(edf_from_solution(p, r, mp, e_transformed, |i, j| {
            workspace.final_aug_matrix[(i, j)]
        }));
    }

    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

pub(super) fn calculate_edfwithworkspace_with_penalty(
    penalized_hessian: &Array2<f64>,
    penalty: &PirlsPenalty,
    workspace: &mut PirlsWorkspace,
) -> Result<f64, EstimationError> {
    match penalty {
        PirlsPenalty::Dense { e_transformed, .. } => {
            calculate_edfwithworkspace(penalized_hessian, e_transformed, workspace)
        }
        PirlsPenalty::Diagonal {
            diag,
            positive_indices,
            ..
        } => calculate_edfwithworkspace_from_diagonal_penalty(
            penalized_hessian,
            diag,
            positive_indices,
            workspace,
        ),
    }
}

pub(super) fn calculate_edf_from_diagonal_penalty(
    penalized_hessian: &SymmetricMatrix,
    diag: &Array1<f64>,
    positive_indices: &[usize],
) -> Result<f64, EstimationError> {
    let p = penalized_hessian.ncols();
    let r = positive_indices.len();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    let mut rhs_arr = Array2::<f64>::zeros((p, r));
    for (col, &idx) in positive_indices.iter().enumerate() {
        rhs_arr[[idx, col]] = 1.0;
    }
    let factor =
        penalized_hessian
            .factorize()
            .map_err(|_| EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
    let sol = factor
        .solvemulti(&rhs_arr)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    let mut tr = 0.0;
    for (col, &idx) in positive_indices.iter().enumerate() {
        tr += diag[idx] * sol[[idx, col]];
    }
    Ok((p as f64 - tr).clamp(mp, p as f64))
}

pub(super) fn calculate_edfwithworkspace_from_diagonal_penalty(
    penalized_hessian: &Array2<f64>,
    diag: &Array1<f64>,
    positive_indices: &[usize],
    workspace: &mut PirlsWorkspace,
) -> Result<f64, EstimationError> {
    let p = penalized_hessian.ncols();
    let r = positive_indices.len();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    if workspace.final_aug_matrix.nrows() != p || workspace.final_aug_matrix.ncols() != r {
        workspace.final_aug_matrix = Array2::zeros((p, r));
    } else {
        workspace.final_aug_matrix.fill(0.0);
    }
    for (col, &idx) in positive_indices.iter().enumerate() {
        workspace.final_aug_matrix[[idx, col]] = 1.0;
    }

    let factor = StableSolver::new("pirls diagonal edf workspace")
        .factorize(penalized_hessian)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    {
        let mut rhsview = array2_to_matmut(&mut workspace.final_aug_matrix);
        factor.solve_in_place(rhsview.as_mut());
    }
    let mut tr = 0.0;
    for (col, &idx) in positive_indices.iter().enumerate() {
        tr += diag[idx] * workspace.final_aug_matrix[[idx, col]];
    }
    Ok((p as f64 - tr).clamp(mp, p as f64))
}

#[inline]
pub(super) fn edf_from_solution<F>(
    p: usize,
    r: usize,
    mp: f64,
    e_transformed: &Array2<f64>,
    solved_at: F,
) -> f64
where
    F: Fn(usize, usize) -> f64,
{
    let mut tr = 0.0;
    for j in 0..r {
        for i in 0..p {
            tr += solved_at(i, j) * e_transformed[(j, i)];
        }
    }
    (p as f64 - tr).clamp(mp, p as f64)
}
