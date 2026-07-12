use crate::estimate::EstimationError;
use gam_linalg::faer_ndarray::{FaerSymmetricFactor, array2_to_matmut};
use gam_linalg::utils::{StableSolver, array_is_finite};
use gam_linalg::matrix::SymmetricMatrix;
use gam_problem::Coefficients;
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
            let mp = (p as f64 - r as f64).max(0.0);
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
            let mp = (p as f64 - r as f64).max(0.0);
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
    factor: &gam_linalg::sparse_exact::SparseExactFactor,
    penalty: &PirlsPenalty,
) -> Result<f64, EstimationError> {
    let PirlsPenalty::Dense { e_transformed, .. } = penalty else {
        crate::bail_invalid_estim!("calculate_edf_from_sparse_factor requires PirlsPenalty::Dense");
    };
    // e_transformed has shape (r, p) — cols give the coefficient dimension p.
    let p = e_transformed.ncols();
    let r = e_transformed.nrows();
    let mp = (p as f64 - r as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    let rhs_arr = e_transformed.t().to_owned();
    let sol =
        gam_linalg::sparse_exact::solve_sparse_spdmulti(factor, &rhs_arr).map_err(|_| {
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
    let mp = (p as f64 - r as f64).max(0.0);
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
    let mp = (p as f64 - r as f64).max(0.0);
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
    let mp = (p as f64 - r as f64).max(0.0);
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
    let mp = (p as f64 - r as f64).max(0.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use gam_linalg::matrix::SymmetricMatrix;
    use ndarray::array;

    /// Regression: a penalty with MORE rows than coefficient columns (`r > p`)
    /// is legitimate for factor-smooth / random-slope / random-effect
    /// structures whose penalty roots are stacked or full-rank. The
    /// min-penalty-dof floor `mp = max(p - r, 0)` must be computed in `f64`,
    /// because the `usize` subtraction `p - r` underflows and panics with
    /// "attempt to subtract with overflow" when `r > p`. This exercises the
    /// `r > p` path on the dense EDF entry point and asserts that the floor is
    /// honored (no panic, finite EDF in `[0, p]`).
    #[test]
    pub(crate) fn calculate_edf_floors_when_penalty_rank_exceeds_coefficient_dim() {
        // p = 2 coefficients, r = 3 penalty rows (r > p).
        let p = 2usize;
        // SPD penalized Hessian (well-conditioned, dense path).
        let hessian = SymmetricMatrix::Dense(array![[4.0, 1.0], [1.0, 3.0]]);
        // e_transformed has shape (r, p) = (3, 2): more penalty rows than
        // coefficient columns — the factor/random-slope structure.
        let e_transformed = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        assert_eq!(e_transformed.nrows(), 3);
        assert_eq!(e_transformed.ncols(), p);

        let edf = calculate_edf(&hessian, &e_transformed)
            .expect("EDF solve should succeed for an SPD Hessian with r > p");

        // mp = max(p - r, 0) = 0, so the EDF is floored at 0 and capped at p.
        assert!(
            edf.is_finite(),
            "EDF must be finite for r > p penalty, got {edf}"
        );
        assert!(
            (0.0..=p as f64).contains(&edf),
            "EDF must lie in [0, {p}] for r > p penalty, got {edf}"
        );
    }
}
