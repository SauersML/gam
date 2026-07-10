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
//! ## Inner-solve integration
//!
//! `inner_blockwise_fit` and the joint-Newton kernels in `custom_family`
//! consume ordinary block-local penalties as a `&[Array2<f64>]` paired with
//! per-block `(start, end)` ranges:
//!
//! * `apply_joint_block_penalty_into(ranges, s_lambdas, …)` (≈ line 19960)
//! * `joint_penalty_preconditioner_diag(…)` (≈ line 20067)
//! * `add_joint_penalty_to_matrix(matrix, ranges, s_lambdas, …)` (≈ line 20132)
//!
//! A cross-block dense `S` has no single owning block range, so the solver also
//! threads a `JointPenaltyBundle` through those helpers as a full-width path
//! that:
//!
//! 1. computes `S · v` as a full `total × total` mat-vec (cf. `fast_av`),
//! 2. accumulates `diag(S)` into the Jacobi preconditioner over the full
//!    parameter vector, and
//! 3. adds `λ · S` to the dense joint Hessian without slicing.
//!
//! The remaining construction-site work is to produce the correct
//! `JointPenaltySpec` instances for each coupled-family compile path; once a
//! bundle is supplied through `BlockwiseFitOptions::joint_penalties`, the inner
//! solve consumes its objective, mat-vec, preconditioner, and dense-Hessian
//! contributions.

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
    NotPositiveSemidefinite {
        min_eigenvalue: f64,
        max_abs_eigenvalue: f64,
    },
    NullspaceMismatch {
        declared: usize,
        numerical: usize,
    },
    EigendecompositionFailed {
        reason: String,
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
            Self::NotPositiveSemidefinite {
                min_eigenvalue,
                max_abs_eigenvalue,
            } => write!(
                f,
                "joint penalty matrix is not positive semidefinite: min eigenvalue \
                 {min_eigenvalue:.6e} (max |eigenvalue| {max_abs_eigenvalue:.6e}); the \
                 penalized objective is unbounded below along the negative mode"
            ),
            Self::NullspaceMismatch {
                declared,
                numerical,
            } => write!(
                f,
                "joint penalty declares nullspace_dim={declared} but the eigenspectrum has \
                 {numerical} numerical-zero direction(s); the REML pseudo-logdet rank would \
                 be wrong"
            ),
            Self::EigendecompositionFailed { reason } => write!(
                f,
                "joint penalty eigendecomposition failed during validation: {reason}"
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
        // PSD + declared-nullity honesty. An indefinite joint penalty makes
        // the penalized objective unbounded below along its negative mode
        // while the pseudo-logdet's positive-eigenspace filter would silently
        // drop that mode; a wrong declared nullity mis-ranks the REML
        // pseudo-logdet (the whole point of declaring it is to avoid runtime
        // thresholds, so it must agree with the spectrum at construction).
        if nrows > 0 {
            use gam_linalg::faer_ndarray::FaerEigh;
            let (eigenvalues, _) =
                FaerEigh::eigh(&self.matrix, faer::Side::Lower).map_err(|e| {
                    JointPenaltyError::EigendecompositionFailed {
                        reason: e.to_string(),
                    }
                })?;
            let max_abs_eigenvalue = eigenvalues
                .iter()
                .fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
            // Same relative classification as the REML pseudo-logdet kernel:
            // the eigensolver noise floor is O(p·ε·‖S‖), never an absolute cut.
            let tol = 100.0 * (nrows as f64) * f64::EPSILON * max_abs_eigenvalue;
            if let Some(&min_eigenvalue) = eigenvalues
                .iter()
                .filter(|&&ev| ev < -tol)
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                return Err(JointPenaltyError::NotPositiveSemidefinite {
                    min_eigenvalue,
                    max_abs_eigenvalue,
                });
            }
            let numerical = eigenvalues.iter().filter(|&&ev| ev <= tol).count();
            if numerical != self.nullspace_dim {
                return Err(JointPenaltyError::NullspaceMismatch {
                    declared: self.nullspace_dim,
                    numerical,
                });
            }
        }
        Ok(())
    }
}

/// Per-evaluation bundle of cross-block penalties paired with their current
/// log-smoothing parameters.
///
/// The outer optimizer concatenates joint penalty `log λ` values onto the
/// per-block ρ vector; the inner solver receives this bundle via
/// [`crate::families::custom_family::BlockwiseFitOptions::joint_penalties`]
/// and adds the full-width quadratic / matvec / preconditioner / Hessian
/// contributions to the joint-Newton primitives.
#[derive(Clone, Debug)]
pub struct JointPenaltyBundle {
    pub specs: std::sync::Arc<Vec<JointPenaltySpec>>,
    pub log_lambdas: Vec<f64>,
}

impl JointPenaltyBundle {
    /// Build a bundle, validating the per-penalty `log λ` count and dimension
    /// agreement against `total_compiled`.
    pub fn new(
        specs: std::sync::Arc<Vec<JointPenaltySpec>>,
        log_lambdas: Vec<f64>,
        total_compiled: usize,
    ) -> Result<Self, String> {
        if specs.len() != log_lambdas.len() {
            return Err(format!(
                "joint penalty bundle: {} specs vs {} log_lambdas",
                specs.len(),
                log_lambdas.len(),
            ));
        }
        for (i, spec) in specs.iter().enumerate() {
            if spec.dim() != total_compiled {
                return Err(format!(
                    "joint penalty {i}: dim {} != total_compiled {}",
                    spec.dim(),
                    total_compiled,
                ));
            }
        }
        Ok(Self { specs, log_lambdas })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Total joint-penalty contribution to the objective:
    ///   `½ Σ_j exp(ρ_j) · βᵀ S_j β`.
    pub fn quadratic(&self, beta: ArrayView1<'_, f64>) -> f64 {
        let mut total = 0.0;
        for (spec, &log_lambda) in self.specs.iter().zip(self.log_lambdas.iter()) {
            let lam = log_lambda.exp();
            total += 0.5 * lam * spec.quadratic_form(beta);
        }
        total
    }

    /// Accumulate `Σ_j exp(ρ_j) · S_j · v` into `out` (additive).
    pub fn add_apply_into(&self, vector: ArrayView1<'_, f64>, out: &mut ndarray::Array1<f64>) {
        assert_eq!(out.len(), vector.len());
        for (spec, &log_lambda) in self.specs.iter().zip(self.log_lambdas.iter()) {
            let lam = log_lambda.exp();
            let sv = spec.matrix.dot(&vector);
            out.scaled_add(lam, &sv);
        }
    }

    /// Accumulate `Σ_j exp(ρ_j) · diag(S_j)` into `diag` (additive).
    pub fn add_diag(&self, diag: &mut ndarray::Array1<f64>) {
        for (spec, &log_lambda) in self.specs.iter().zip(self.log_lambdas.iter()) {
            let lam = log_lambda.exp();
            for (i, value) in spec.matrix.diag().iter().enumerate() {
                diag[i] += lam * *value;
            }
        }
    }

    /// Accumulate `Σ_j exp(ρ_j) · S_j` into the full `matrix` (additive).
    pub fn add_to_matrix(&self, matrix: &mut Array2<f64>) {
        assert_eq!(matrix.nrows(), matrix.ncols());
        for (spec, &log_lambda) in self.specs.iter().zip(self.log_lambdas.iter()) {
            let lam = log_lambda.exp();
            matrix.scaled_add(lam, &spec.matrix);
        }
    }

    /// Per-penalty ρ-gradient contribution to the outer objective term:
    ///   `∂/∂ρ_j [½ exp(ρ_j) βᵀ S_j β] = exp(ρ_j) · ½ βᵀ S_j β`.
    pub fn rho_objective_gradient(&self, beta: ArrayView1<'_, f64>, out: &mut [f64]) {
        assert_eq!(out.len(), self.specs.len());
        for (i, (spec, &log_lambda)) in self.specs.iter().zip(self.log_lambdas.iter()).enumerate() {
            let lam = log_lambda.exp();
            out[i] = 0.5 * lam * spec.quadratic_form(beta);
        }
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
        let result = cross_block_spec().validate();
        assert!(
            result.is_ok(),
            "valid cross-block spec rejected: {result:?}"
        );
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
        use gam_linalg::faer_ndarray::FaerEigh;
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

    /// 2-block toy with one full-width SPD joint penalty:
    ///
    /// Two scalar blocks (`p = 1 + 1 = 2`). The unpenalised "log-likelihood"
    /// is a quadratic with optimum at `b`:
    ///     ℓ(β) = −½ (β − b)ᵀ I (β − b)
    /// so `−∇ℓ = (β − b)` and `−∇²ℓ = I`. We add ONE joint penalty
    /// `S = [[2, 1], [1, 2]]` (SPD, full rank, cross-block coupling
    /// off-diagonal). With `λ = exp(ρ)` the penalised objective is
    ///     F(β) = ½ (β − b)ᵀ (β − b) + ½ λ βᵀ S β
    /// whose minimiser solves `(I + λ S) β̂ = b`. We verify the bundle's
    /// `add_to_matrix` builds the right LHS and the `add_apply_into` /
    /// `quadratic` helpers agree with the analytic gradient / objective at
    /// `β̂`.
    #[test]
    fn bundle_two_block_minimiser_matches_analytic_solution() {
        use gam_linalg::faer_ndarray::FaerCholesky;
        use ndarray::Array2;

        let spec = JointPenaltySpec {
            label: Some("toy_cross_block".to_string()),
            matrix: array![[2.0_f64, 1.0], [1.0, 2.0]],
            initial_log_lambda: 0.0,
            nullspace_dim: 0,
        };
        let log_lambda = -0.4_f64;
        let lam = log_lambda.exp();
        let bundle = JointPenaltyBundle::new(std::sync::Arc::new(vec![spec]), vec![log_lambda], 2)
            .expect("valid bundle");

        // Build LHS = I + λ S via add_to_matrix (the exact path the inner
        // Newton uses to assemble the penalised joint Hessian).
        let mut lhs = Array2::<f64>::eye(2);
        bundle.add_to_matrix(&mut lhs);
        // Verify add_to_matrix produced I + λ S.
        let expected_lhs = array![[1.0 + lam * 2.0, lam], [lam, 1.0 + lam * 2.0]];
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (lhs[[r, c]] - expected_lhs[[r, c]]).abs() < 1e-12,
                    "lhs[{r}, {c}] = {} expected {}",
                    lhs[[r, c]],
                    expected_lhs[[r, c]]
                );
            }
        }

        // Solve (I + λ S) β̂ = b for b = [1.0, -0.5].
        let b: Array1<f64> = array![1.0, -0.5];
        let chol = lhs.cholesky(faer::Side::Lower).expect("SPD");
        let mut rhs_mat = Array2::<f64>::zeros((2, 1));
        rhs_mat[[0, 0]] = b[0];
        rhs_mat[[1, 0]] = b[1];
        let mut beta_mat = rhs_mat.clone();
        chol.solve_mat_in_place(&mut beta_mat);
        let beta_hat: Array1<f64> = array![beta_mat[[0, 0]], beta_mat[[1, 0]]];

        // Gradient at β̂: (β̂ − b) + λ S β̂ should be ~0.
        let mut grad = &beta_hat - &b;
        bundle.add_apply_into(beta_hat.view(), &mut grad);
        let grad_inf = grad.iter().map(|v: &f64| v.abs()).fold(0.0_f64, f64::max);
        assert!(
            grad_inf < 1e-12,
            "penalised gradient at analytic minimiser must vanish: {grad_inf:.3e}"
        );

        // Objective ½(β̂−b)·(β̂−b) + bundle.quadratic(β̂) reproduces the
        // closed-form minimum value F(β̂) = ½(β̂−b)ᵀ(β̂−b) + ½λ β̂ᵀ S β̂.
        let resid = &beta_hat - &b;
        let unpen = 0.5 * resid.dot(&resid);
        let pen = bundle.quadratic(beta_hat.view());
        let expected_obj = 0.5 * resid.dot(&resid)
            + 0.5 * lam * beta_hat.dot(&array![[2.0, 1.0], [1.0, 2.0]].dot(&beta_hat));
        assert!(
            (unpen + pen - expected_obj).abs() < 1e-12,
            "objective sum {} mismatched expected {}",
            unpen + pen,
            expected_obj
        );

        // Preconditioner diag accumulator: diag(I) + λ diag(S) = [1+2λ, 1+2λ].
        let mut diag = ndarray::Array1::<f64>::from_elem(2, 1.0);
        bundle.add_diag(&mut diag);
        assert!((diag[0] - (1.0 + lam * 2.0)).abs() < 1e-12);
        assert!((diag[1] - (1.0 + lam * 2.0)).abs() < 1e-12);

        // rho-objective-gradient: ½ λ β̂ᵀ S β̂.
        let mut rho_grad = vec![0.0_f64];
        bundle.rho_objective_gradient(beta_hat.view(), &mut rho_grad);
        let expected_rho_grad =
            0.5 * lam * beta_hat.dot(&array![[2.0, 1.0], [1.0, 2.0]].dot(&beta_hat));
        assert!(
            (rho_grad[0] - expected_rho_grad).abs() < 1e-12,
            "rho-grad {} expected {}",
            rho_grad[0],
            expected_rho_grad
        );
    }

    #[test]
    fn bundle_rejects_dim_mismatch() {
        let spec = JointPenaltySpec {
            label: None,
            matrix: Array2::<f64>::eye(3),
            initial_log_lambda: 0.0,
            nullspace_dim: 0,
        };
        let err = JointPenaltyBundle::new(std::sync::Arc::new(vec![spec]), vec![0.0], 4)
            .expect_err("dim mismatch must reject");
        assert!(err.contains("total_compiled"));
    }

    #[test]
    fn bundle_rejects_lambda_count_mismatch() {
        let spec = JointPenaltySpec {
            label: None,
            matrix: Array2::<f64>::eye(2),
            initial_log_lambda: 0.0,
            nullspace_dim: 0,
        };
        let err = JointPenaltyBundle::new(std::sync::Arc::new(vec![spec]), vec![], 2)
            .expect_err("count mismatch must reject");
        assert!(err.contains("specs vs"));
    }
}
