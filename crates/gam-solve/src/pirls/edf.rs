use crate::estimate::EstimationError;
use gam_linalg::faer_ndarray::{FaerSymmetricFactor, array2_to_matmut};
use gam_linalg::matrix::SymmetricMatrix;
use gam_linalg::utils::{StableSolver, array_is_finite};
use gam_problem::Coefficients;
use ndarray::Array2;

use super::{PirlsPenalty, PirlsWorkspace};

/// Result of the stable penalized least squares solve
#[derive(Clone)]
pub(crate) struct StablePLSResult {
    /// Solution vector beta
    pub beta: Coefficients,
    /// Final penalized Hessian matrix (sparse or dense depending on solve path)
    pub penalized_hessian: SymmetricMatrix,
    /// Effective degrees of freedom
    pub edf: f64,
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
                return edf_from_solution(p, r, mp, e_transformed, |i, j| {
                    workspace.final_aug_matrix[(i, j)]
                });
            }
            Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })
        }
    }
}

/// EDF from an already-factorized sparse penalized Hessian (sparse path).
///
/// Mirrors `calculate_edf_with_penalty` but accepts the `SparseExactFactor`
/// that PLS already produced, eliminating the redundant second sparse
/// factorization inside every PIRLS outer iteration.
///
/// `tr(H⁻¹ S_λ) = tr(H⁻¹ EᵀE)` is exact either way it is formed: as
/// `Σ_r e_r H⁻¹ e_rᵀ` against the Takahashi selected inverse (every pair inside
/// a row's support is a nonzero of `S_λ ⊆ H`, so on the selected pattern), or by
/// solving `H X = Eᵀ` for all `r` columns. The first costs the recurrence
/// `Σ_j c_j²` over `L`'s column counts, the second `4·nnz(L)·r`; the cheaper one
/// is taken. For a many-level random effect `r` is the level count and the
/// recurrence is linear in it, where the solves are quadratic.
pub(super) fn calculate_edf_from_sparse_factor(
    factor: &gam_linalg::sparse_exact::SparseExactFactor,
    penalty: &PirlsPenalty,
) -> Result<f64, EstimationError> {
    let PirlsPenalty::Dense { e_transformed, .. } = penalty;
    // e_transformed has shape (r, p) — cols give the coefficient dimension p.
    let p = e_transformed.ncols();
    let r = e_transformed.nrows();
    let mp = (p as f64 - r as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    let ill_conditioned = || EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    };
    let solve_flops = 4usize.saturating_mul(factor.factor_nnz()).saturating_mul(r);
    if factor.selected_inverse_flops() < solve_flops {
        let taka = factor.selected_inverse().map_err(|_| ill_conditioned())?;
        let tr = taka.trace_root_gram(e_transformed.view(), 0);
        if tr.is_finite() {
            return Ok((p as f64 - tr).clamp(mp, p as f64));
        }
        return Err(ill_conditioned());
    }
    let rhs_arr = e_transformed.t().to_owned();
    let sol = gam_linalg::sparse_exact::solve_sparse_spdmulti(factor, &rhs_arr)
        .map_err(|_| ill_conditioned())?;
    if sol.nrows() == p && sol.ncols() == r && sol.iter().all(|v| v.is_finite()) {
        return edf_from_solution(p, r, mp, e_transformed, |i, j| {
            sol[[i, j]]
        });
    }
    Err(ill_conditioned())
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
        return edf_from_solution(p, r, mp, e_transformed, |i, j| {
            sol[[i, j]]
        });
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

    let factor = StableSolver::new()
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
        return edf_from_solution(p, r, mp, e_transformed, |i, j| {
            workspace.final_aug_matrix[(i, j)]
        });
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
    }
}

#[inline]
pub(super) fn edf_from_solution<F>(
    p: usize,
    r: usize,
    mp: f64,
    e_transformed: &Array2<f64>,
    solved_at: F,
) -> Result<f64, EstimationError>
where
    F: Fn(usize, usize) -> f64,
{
    let mut tr = 0.0;
    for j in 0..r {
        for i in 0..p {
            tr += solved_at(i, j) * e_transformed[(j, i)];
        }
    }
    // `clamp` passes NaN through, so a non-finite trace must be refused here:
    // there is no EDF to report when tr(H⁻¹S) is not a number.
    if !tr.is_finite() {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    Ok((p as f64 - tr).clamp(mp, p as f64))
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

    /// The sparse-factor EDF, in both of its exact forms (selected-inverse
    /// contraction and multi-RHS solve), equals the dense-solve EDF on a
    /// random-effect-shaped system: a ridge root over `levels` indicator
    /// columns next to a dense smooth block.
    #[test]
    fn sparse_factor_edf_matches_dense_edf() {
        use faer::sparse::{SparseColMat, Triplet};
        use gam_linalg::sparse_exact::{factorize_sparse_spd, solve_sparse_spdmulti};

        let levels = 9usize;
        let smooth = 3usize;
        let p = levels + smooth;
        let n = 60usize;
        let mut x = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            x[[i, i % levels]] = 1.0;
            let t = i as f64 / n as f64;
            x[[i, levels]] = t;
            x[[i, levels + 1]] = (3.0 * t).sin();
            x[[i, levels + 2]] = t * t - 0.3;
        }
        // Ridge root on the levels, second-difference-like root on the smooth.
        let r = levels + 2;
        let mut e = Array2::<f64>::zeros((r, p));
        for l in 0..levels {
            e[[l, l]] = 0.7 + 0.1 * l as f64;
        }
        e[[levels, levels]] = 1.3;
        e[[levels, levels + 1]] = -2.6;
        e[[levels, levels + 2]] = 1.3;
        e[[levels + 1, levels + 1]] = 0.9;
        e[[levels + 1, levels + 2]] = -0.9;
        let s = e.t().dot(&e);
        let h = x.t().dot(&x) + &s;

        let triplets: Vec<Triplet<usize, usize, f64>> = (0..p)
            .flat_map(|j| (0..=j).map(move |i| (i, j)))
            .filter(|&(i, j)| h[[i, j]] != 0.0)
            .map(|(i, j)| Triplet::new(i, j, h[[i, j]]))
            .collect();
        let h_sparse = SparseColMat::try_new_from_triplets(p, p, &triplets).unwrap();
        let factor = factorize_sparse_spd(&h_sparse).unwrap();

        let dense_edf = calculate_edf(&SymmetricMatrix::Dense(h.clone()), &e).unwrap();
        let penalty = PirlsPenalty::Dense {
            s_transformed: s,
            e_transformed: e.clone(),
        };
        let sparse_edf = calculate_edf_from_sparse_factor(&factor, &penalty).unwrap();
        let tol = 1e-12 * p as f64;
        assert!(
            (sparse_edf - dense_edf).abs() <= tol,
            "sparse-factor EDF {sparse_edf} vs dense {dense_edf}"
        );

        let taka = factor.selected_inverse().unwrap();
        let trace_selected = taka.trace_root_gram(e.view(), 0);
        let sol = solve_sparse_spdmulti(&factor, &e.t().to_owned()).unwrap();
        let trace_solved: f64 = (0..r)
            .map(|k| (0..p).map(|i| sol[[i, k]] * e[[k, i]]).sum::<f64>())
            .sum();
        assert!(
            (trace_selected - trace_solved).abs() <= tol,
            "selected-inverse trace {trace_selected} vs solved {trace_solved}"
        );
        assert!(
            (p as f64 - trace_selected - dense_edf).abs() <= tol,
            "selected-inverse EDF {} vs dense {dense_edf}",
            p as f64 - trace_selected
        );
    }
}
