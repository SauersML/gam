//! Joint (cross-block) penalty specifications.
//!
//! After the `T^T S_j T` pullback used by the V+M / SMGS-exact compile path,
//! a single penalty `S_j` no longer has its nonzero region confined to one
//! `ParameterBlockSpec`: the pullback by the inter-block coupling matrix `T`
//! distributes weight across the *entire* compiled parameter vector. The
//! existing `ParameterBlockSpec.penalties: Vec<PenaltyMatrix>` model encodes
//! a per-block-local penalty (its dim equals the owning block's column count),
//! so it cannot represent these full-width operators.
//!
//! [`JointPenaltySpec`] is the carrier for that case: one dense
//! `total_compiled × total_compiled` matrix with its own initial smoothing
//! parameter and structural nullspace dimension. It lives *alongside*, not
//! *inside*, the per-block specs.
//!
//! ## Inner-solve integration point (NOT wired here)
//!
//! `inner_blockwise_fit` and the joint-Newton kernels in
//! `custom_family.rs` consume penalties as a `&[Array2<f64>]` paired with
//! per-block `(start, end)` ranges:
//!
//! * `apply_joint_block_penalty_into(ranges, s_lambdas, …)` (≈ line 19960)
//! * `joint_penalty_preconditioner_diag(…)` (≈ line 20067)
//! * `add_joint_penalty_to_matrix(matrix, ranges, s_lambdas, …)` (≈ line 20132)
//!
//! All three iterate `s_lambdas[b]` and write into the slice
//! `matrix[ranges[b].0..ranges[b].1, ranges[b].0..ranges[b].1]`. A
//! cross-block dense `S` has no such single range, so wiring a
//! `JointPenaltySpec` requires an additional "full-width" path that:
//!
//! 1. computes `S · v` as a full `total × total` mat-vec (cf. `fast_av`),
//! 2. accumulates `diag(S)` into the Jacobi preconditioner over the full
//!    parameter vector, and
//! 3. adds `λ · S` to the dense joint Hessian without slicing.
//!
//! That wiring is intentionally out of scope for this module: the type is
//! published first so other agents can build the construction-site, REML
//! pseudo-logdet, and inner-solve hookup against a stable target.

use ndarray::{Array2, ArrayView1};

/// A penalty whose support spans the entire compiled parameter vector.
///
/// Unlike [`crate::families::custom_family::PenaltyMatrix`], this carries a
/// single dense `total_compiled × total_compiled` quadratic form — the
/// shape produced by `T^T S_j T` pullback after the V+M / SMGS-exact
/// compile. The `nullspace_dim` is the structural dimension of `ker(S)`
/// as reported by the construction site (rank-revealing on the *pulled-back*
/// operator, not the pre-pullback `S_j`), so the REML pseudo-logdet can
/// avoid numerical rank thresholds.
#[derive(Debug, Clone)]
pub struct JointPenaltySpec {
    /// Optional user-visible precision label. Joint penalties that share a
    /// label share one smoothing parameter (same convention as
    /// [`crate::families::custom_family::PenaltyMatrix::Labeled`]).
    pub label: Option<String>,
    /// Dense symmetric PSD matrix of shape `(total_compiled, total_compiled)`.
    pub matrix: Array2<f64>,
    /// Initial value of `log λ` for this penalty.
    pub initial_log_lambda: f64,
    /// Structural nullspace dimension of `matrix` (i.e. `total_compiled - rank`).
    pub nullspace_dim: usize,
}

/// Reason a [`JointPenaltySpec`] failed validation.
#[derive(Debug, Clone, PartialEq)]
pub enum JointPenaltyError {
    NotSquare {
        nrows: usize,
        ncols: usize,
    },
    NonFiniteEntry {
        row: usize,
        col: usize,
        value: f64,
    },
    NonFiniteInitialLogLambda {
        value: f64,
    },
    NotSymmetric {
        row: usize,
        col: usize,
        asymmetry: f64,
    },
    NullspaceTooLarge {
        total: usize,
        nullspace_dim: usize,
    },
}

impl std::fmt::Display for JointPenaltyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSquare { nrows, ncols } => {
                write!(f, "joint penalty matrix is not square: {nrows}x{ncols}")
            }
            Self::NonFiniteEntry { row, col, value } => write!(
                f,
                "joint penalty matrix has non-finite entry at ({row},{col}): {value}"
            ),
            Self::NonFiniteInitialLogLambda { value } => {
                write!(f, "joint penalty initial_log_lambda is non-finite: {value}")
            }
            Self::NotSymmetric {
                row,
                col,
                asymmetry,
            } => write!(
                f,
                "joint penalty matrix is not symmetric at ({row},{col}): |S - Sᵀ|={asymmetry:.3e}"
            ),
            Self::NullspaceTooLarge {
                total,
                nullspace_dim,
            } => write!(
                f,
                "joint penalty nullspace_dim={nullspace_dim} exceeds dim={total}"
            ),
        }
    }
}

impl std::error::Error for JointPenaltyError {}

impl JointPenaltySpec {
    /// Symmetry tolerance for [`validate`]. Cross-block pullbacks via `T`
    /// accumulate roundoff, so an exact symmetric requirement is too tight;
    /// this matches the floor used by the surrounding penalty code paths.
    const SYMMETRY_TOL: f64 = 1e-10;

    /// Total compiled parameter count this penalty acts on.
    #[inline]
    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }

    /// Trace of the penalty matrix (`Σ_i S[i,i]`).
    pub fn trace(&self) -> f64 {
        self.matrix.diag().iter().copied().sum()
    }

    /// Structural pseudo-rank, derived from the declared `nullspace_dim`.
    /// This is the rank used by the REML pseudo-logdet under the
    /// no-numerical-thresholds policy in the surrounding code.
    #[inline]
    pub fn pseudo_rank(&self) -> usize {
        self.dim().saturating_sub(self.nullspace_dim)
    }

    /// Quadratic form `βᵀ S β`. Mirrors
    /// [`crate::families::custom_family::PenaltyMatrix::quadratic_form`] for
    /// the full-width case.
    pub fn quadratic_form(&self, beta: ArrayView1<'_, f64>) -> f64 {
        assert_eq!(
            beta.len(),
            self.dim(),
            "joint penalty quadratic form: beta length {} != dim {}",
            beta.len(),
            self.dim()
        );
        beta.dot(&self.matrix.dot(&beta))
    }

    /// Validate shape, finiteness, symmetry, and nullspace bookkeeping.
    pub fn validate(&self) -> Result<(), JointPenaltyError> {
        let (nrows, ncols) = self.matrix.dim();
        if nrows != ncols {
            return Err(JointPenaltyError::NotSquare { nrows, ncols });
        }
        if !self.initial_log_lambda.is_finite() {
            return Err(JointPenaltyError::NonFiniteInitialLogLambda {
                value: self.initial_log_lambda,
            });
        }
        if self.nullspace_dim > nrows {
            return Err(JointPenaltyError::NullspaceTooLarge {
                total: nrows,
                nullspace_dim: self.nullspace_dim,
            });
        }
        for ((row, col), &value) in self.matrix.indexed_iter() {
            if !value.is_finite() {
                return Err(JointPenaltyError::NonFiniteEntry { row, col, value });
            }
        }
        for row in 0..nrows {
            for col in (row + 1)..ncols {
                let asymmetry = (self.matrix[[row, col]] - self.matrix[[col, row]]).abs();
                if asymmetry > Self::SYMMETRY_TOL {
                    return Err(JointPenaltyError::NotSymmetric {
                        row,
                        col,
                        asymmetry,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, array};

    /// 4-dim cross-block dense penalty: a rank-2 operator that couples
    /// indices {0,1} to {2,3} (i.e. nonzero off the 2×2 block diagonal),
    /// which is exactly the shape that defeats a per-block `PenaltyMatrix`.
    fn cross_block_spec() -> JointPenaltySpec {
        // Build S = vᵀv + wᵀw where v and w span across both 2-blocks.
        let v: Array1<f64> = array![1.0, 0.0, -1.0, 0.0];
        let w: Array1<f64> = array![0.0, 1.0, 0.0, -1.0];
        let mut matrix: Array2<f64> = Array2::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                matrix[[i, j]] = v[i] * v[j] + w[i] * w[j];
            }
        }
        JointPenaltySpec {
            label: Some("cross_block_pullback".to_string()),
            matrix,
            initial_log_lambda: -1.5,
            nullspace_dim: 2,
        }
    }

    #[test]
    fn cross_block_dense_validates() {
        cross_block_spec().validate().expect("valid cross-block spec");
    }

    #[test]
    fn trace_matches_diagonal_sum() {
        let spec = cross_block_spec();
        // diag(S) = [v0^2+w0^2, v1^2+w1^2, v2^2+w2^2, v3^2+w3^2] = [1,1,1,1]
        assert!((spec.trace() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn pseudo_rank_uses_declared_nullspace() {
        let spec = cross_block_spec();
        assert_eq!(spec.dim(), 4);
        assert_eq!(spec.pseudo_rank(), 2);
    }

    #[test]
    fn quadratic_form_matches_explicit_mat_vec() {
        let spec = cross_block_spec();
        // Pick a beta that has support in both 2-blocks.
        let beta: Array1<f64> = array![0.5, -0.25, 1.0, 0.75];
        // v·β = 0.5 - 1.0 = -0.5; w·β = -0.25 - 0.75 = -1.0
        // βᵀSβ = (v·β)^2 + (w·β)^2 = 0.25 + 1.0 = 1.25
        let q = spec.quadratic_form(beta.view());
        assert!((q - 1.25).abs() < 1e-12, "got {q}");
    }

    #[test]
    fn determinant_zero_for_rank_deficient_matches_nullspace() {
        use crate::faer_ndarray::FaerEigh;
        let spec = cross_block_spec();
        // Symmetric eigendecomposition; expect exactly nullspace_dim
        // zeros (up to floating-point), matching the declared rank.
        let (eigvals, _) =
            FaerEigh::eigh(&spec.matrix, faer::Side::Lower).expect("symmetric eigh succeeds");
        let mut sorted: Vec<f64> = eigvals.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let zeros = sorted.iter().take_while(|&&v| v.abs() < 1e-10).count();
        assert_eq!(
            zeros, spec.nullspace_dim,
            "spectrum {sorted:?} should have {} near-zeros",
            spec.nullspace_dim
        );
        // Determinant = product of eigenvalues; with a real nullspace
        // it is exactly zero modulo roundoff.
        let det: f64 = sorted.iter().product();
        assert!(det.abs() < 1e-10, "expected ~0 determinant, got {det}");
    }

    #[test]
    fn validate_rejects_non_square() {
        let spec = JointPenaltySpec {
            label: None,
            matrix: Array2::zeros((3, 4)),
            initial_log_lambda: 0.0,
            nullspace_dim: 0,
        };
        assert!(matches!(
            spec.validate(),
            Err(JointPenaltyError::NotSquare { nrows: 3, ncols: 4 })
        ));
    }

    #[test]
    fn validate_rejects_non_symmetric() {
        let mut matrix = Array2::<f64>::zeros((3, 3));
        matrix[[0, 1]] = 1.0;
        matrix[[1, 0]] = -1.0;
        let spec = JointPenaltySpec {
            label: None,
            matrix,
            initial_log_lambda: 0.0,
            nullspace_dim: 0,
        };
        assert!(matches!(
            spec.validate(),
            Err(JointPenaltyError::NotSymmetric { .. })
        ));
    }

    #[test]
    fn validate_rejects_oversized_nullspace() {
        let spec = JointPenaltySpec {
            label: None,
            matrix: Array2::zeros((3, 3)),
            initial_log_lambda: 0.0,
            nullspace_dim: 4,
        };
        assert!(matches!(
            spec.validate(),
            Err(JointPenaltyError::NullspaceTooLarge {
                total: 3,
                nullspace_dim: 4
            })
        ));
    }

    #[test]
    fn validate_rejects_non_finite_initial_log_lambda() {
        let spec = JointPenaltySpec {
            label: None,
            matrix: Array2::zeros((2, 2)),
            initial_log_lambda: f64::NAN,
            nullspace_dim: 0,
        };
        assert!(matches!(
            spec.validate(),
            Err(JointPenaltyError::NonFiniteInitialLogLambda { .. })
        ));
    }
}
