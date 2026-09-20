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
/// Unlike `crate::families::custom_family::PenaltyMatrix`, this carries a
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
    /// `crate::families::custom_family::PenaltyMatrix::Labeled`).
    pub label: Option<String>,
    /// Dense symmetric PSD matrix of shape `(total_compiled, total_compiled)`.
    pub matrix: Array2<f64>,
    /// Initial value of `log λ` for this penalty.
    pub initial_log_lambda: f64,
    /// Structural nullspace dimension of `matrix` (i.e. `total_compiled - rank`).
    pub nullspace_dim: usize,
    /// Optional term grouping, declared by the producing family.
    ///
    /// Specs sharing a group are the SAME smooth term seen through different
    /// class contrasts, so a relabeling permutes them among themselves. Any
    /// consumer that needs a reference-INVARIANT quantity per term must
    /// aggregate over the group rather than read one spec: an individual spec's
    /// matrix is expressed in the stacked ALR basis, whose meaning depends on
    /// which class is the reference (#2579). This is deliberately a declared
    /// integer and not something recovered by parsing [`Self::label`] — a
    /// substring classifier over a formatted name is exactly the failure #2593
    /// was closed for.
    ///
    /// `None` means "stands alone", which is every family that does not group.
    pub group: Option<usize>,
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
    InitialLogStrengthOutOfDomain {
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
            Self::InitialLogStrengthOutOfDomain { value } => {
                write!(
                    f,
                    "joint penalty initial_log_lambda is outside the exact strength domain: {value}"
                )
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

/// The thin root `R` of a symmetric penalty `S = RᵀR` on its structural range
/// (#2954): `R = Σ_+^{1/2}U_+ᵀ` over the eigenpairs `S`'s rank keeps. Every
/// penalty value, gradient and curvature is formed from it, so a penalty and
/// its derivatives are one function, and the rows of `R` exclude the null
/// directions by construction.
///
/// That exclusion is the point. A penalty stored in f64 is not exactly singular
/// on its null space: each formed entry carries rounding, and the stored
/// matrix's null eigenvalues sit at `O(u·‖S‖)`, not at zero (measured on
/// `declared_latent_law_2923`: `σ_null = 9.92e-17 = 0.89·u·‖S‖₂`). A dense
/// `½λβᵀSβ` then carries `½λσ_null(uᵀβ)²`, which grows with `λ` (1.8e-3 at
/// `ρ = 29.78`, log-log slope 1 in `λ`) while the structural value vanishes.
/// Truncating the null eigenpairs removes that term. The root's own eigenvector
/// error leaves `λ(u‖R‖|β|)²`, second order.
///
/// Two things are known about each direction, and each can only remove it from
/// the root:
/// * the spectrum, through the one rank rule
///   [`gam_linalg::roundoff::resolved_eigenvalue_count`]: an eigenvalue at or
///   below the eigensolver's Weyl band `p·ε·‖S‖₂` plus `formation_band` (the
///   caller's bound, in eigenvalue units, on the error its construction left in
///   the stored matrix) is not resolved from zero, so no root carries it
///   faithfully;
/// * a declared nullity, a structural fact about the exact penalty: its null
///   directions are dropped even where the stored matrix resolves them, because
///   a formed matrix carries its construction's rounding there. The link
///   wiggle's order-3 I-spline roughness `CᵀS_BC` stores its second structural
///   zero at `1.687e-11` against a Weyl band of `1.370e-11` (gam#2921's 48-row
///   request): the band omits the congruence's own formation error.
///
/// So the rank is the resolved count, capped by the declared range. A
/// disagreement is not refused: judging a declaration against the spectrum needs
/// the construction's formation band, and every producer passes 0 until it
/// carries one (the one-rank-rule follow-up of #2954, which derives it from the
/// assembly's error-magnitude diagonal, and declared null bases, gam#3023).
/// Undeclared, the resolved count is the rank.
pub fn structural_penalty_root(
    matrix: &Array2<f64>,
    declared_nullity: Option<usize>,
    formation_band: f64,
) -> Result<Array2<f64>, PenaltyRootError> {
    let p = matrix.nrows();
    if matrix.ncols() != p {
        return Err(PenaltyRootError::NotSquare {
            nrows: p,
            ncols: matrix.ncols(),
        });
    }
    if let Some(declared) = declared_nullity
        && declared > p
    {
        return Err(PenaltyRootError::NullityExceedsDimension { dim: p, declared });
    }
    if p == 0 {
        return Ok(Array2::zeros((0, 0)));
    }
    use gam_linalg::faer_ndarray::FaerEigh;
    let (eigenvalues, eigenvectors) = FaerEigh::eigh(matrix, faer::Side::Lower).map_err(|e| {
        PenaltyRootError::EigendecompositionFailed {
            reason: e.to_string(),
        }
    })?;
    Ok(root_on_structural_range(
        &eigenvalues.to_vec(),
        &eigenvectors,
        declared_nullity,
        formation_band,
    ))
}

/// [`structural_penalty_root`] from the spectrum `(eigenvalues, eigenvectors)`
/// the caller already computed.
pub(crate) fn root_on_structural_range(
    eigenvalues: &[f64],
    eigenvectors: &Array2<f64>,
    declared_nullity: Option<usize>,
    formation_band: f64,
) -> Array2<f64> {
    let p = eigenvalues.len();
    let resolved = gam_linalg::roundoff::resolved_eigenvalue_count(eigenvalues, formation_band);
    // The declared range caps the resolved count; a nullity above `p` is
    // refused by the caller before the spectrum is formed.
    let rank = declared_nullity.map_or(resolved, |declared| resolved.min(p - declared));
    let nullity = p - rank;
    // Ascending by value: the structural zeros, and any negative rounding among
    // them, come first, so the kept eigenvalues are the last `rank`.
    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
    let kept = &order[nullity..];
    let mut root = Array2::<f64>::zeros((kept.len(), p));
    for (row, &index) in kept.iter().enumerate() {
        let scale = eigenvalues[index].sqrt();
        for column in 0..p {
            root[[row, column]] = scale * eigenvectors[[column, index]];
        }
    }
    root
}

/// Why [`structural_penalty_root`] could not root a penalty.
#[derive(Clone, Debug, PartialEq)]
pub enum PenaltyRootError {
    NotSquare { nrows: usize, ncols: usize },
    NullityExceedsDimension { dim: usize, declared: usize },
    EigendecompositionFailed { reason: String },
}

impl std::fmt::Display for PenaltyRootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSquare { nrows, ncols } => {
                write!(f, "penalty matrix is not square: {nrows}x{ncols}")
            }
            Self::NullityExceedsDimension { dim, declared } => {
                write!(
                    f,
                    "penalty declares nullity {declared} above its dimension {dim}"
                )
            }
            Self::EigendecompositionFailed { reason } => {
                write!(f, "penalty eigendecomposition failed: {reason}")
            }
        }
    }
}

impl std::error::Error for PenaltyRootError {}

impl JointPenaltySpec {
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
    pub(crate) fn pseudo_rank(&self) -> usize {
        self.dim().saturating_sub(self.nullspace_dim)
    }

    /// Quadratic form `βᵀ S β`. Mirrors
    /// `crate::families::custom_family::PenaltyMatrix::quadratic_form` for
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

    /// Validate shape, finiteness, symmetry, PSD, and nullspace bookkeeping,
    /// returning the invariant thin root `R` such that `matrix = RᵀR`.
    ///
    /// Joint-penalty strengths vary throughout an outer optimization, but the
    /// component matrices do not. Returning the root from the same
    /// eigendecomposition that validates the component lets callers retain it
    /// as penalty geometry instead of repeating one O(p³) decomposition per
    /// component on every objective evaluation.
    pub fn validated_root(&self) -> Result<Array2<f64>, JointPenaltyError> {
        let (nrows, ncols) = self.matrix.dim();
        if nrows != ncols {
            return Err(JointPenaltyError::NotSquare { nrows, ncols });
        }
        if crate::validate_log_strength(self.initial_log_lambda).is_err() {
            return Err(JointPenaltyError::InitialLogStrengthOutOfDomain {
                value: self.initial_log_lambda,
            });
        }
        if self.nullspace_dim > nrows {
            return Err(JointPenaltyError::NullspaceTooLarge {
                total: nrows,
                nullspace_dim: self.nullspace_dim,
            });
        }
        let mut magnitude = 0.0_f64;
        for ((row, col), &value) in self.matrix.indexed_iter() {
            if !value.is_finite() {
                return Err(JointPenaltyError::NonFiniteEntry { row, col, value });
            }
            magnitude = magnitude.max(value.abs());
        }
        // A cross-block pullback `TᵀST` forms each entry from `n²` rounded products,
        // so the two triangles can disagree by that accumulation's rounding band
        // `γ_{n²}·max|S_ij|`; anything larger is not a symmetric penalty.
        let symmetry_band = gam_linalg::roundoff::accumulation_growth(nrows * nrows) * magnitude;
        for row in 0..nrows {
            for col in (row + 1)..ncols {
                let asymmetry = (self.matrix[[row, col]] - self.matrix[[col, row]]).abs();
                if asymmetry > symmetry_band {
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
        if nrows == 0 {
            return Ok(Array2::zeros((0, 0)));
        }
        use gam_linalg::faer_ndarray::FaerEigh;
        let (eigenvalues, eigenvectors) =
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
        let active: Vec<usize> = eigenvalues
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| (value > tol).then_some(index))
            .collect();
        let numerical = nrows - active.len();
        if numerical != self.nullspace_dim {
            return Err(JointPenaltyError::NullspaceMismatch {
                declared: self.nullspace_dim,
                numerical,
            });
        }
        let mut root = Array2::<f64>::zeros((active.len(), nrows));
        for (root_row, &eigen_index) in active.iter().enumerate() {
            let scale = eigenvalues[eigen_index].sqrt();
            for column in 0..nrows {
                root[[root_row, column]] = scale * eigenvectors[[column, eigen_index]];
            }
        }
        Ok(root)
    }

    /// Validate this joint penalty without retaining its spectral root.
    pub fn validate(&self) -> Result<(), JointPenaltyError> {
        self.validated_root().map(|_| ())
    }
}

/// Per-evaluation bundle of cross-block penalties paired with their current
/// log-smoothing parameters.
///
/// The outer optimizer concatenates joint penalty `log λ` values onto the
/// per-block ρ vector; the inner solver receives this bundle via
/// `crate::families::custom_family::BlockwiseFitOptions::joint_penalties`
/// and adds the full-width quadratic / matvec / preconditioner / Hessian
/// contributions to the joint-Newton primitives.
#[derive(Clone, Debug)]
pub struct JointPenaltyBundle {
    specs: std::sync::Arc<Vec<JointPenaltySpec>>,
    roots: std::sync::Arc<Vec<Array2<f64>>>,
    log_lambdas: Vec<f64>,
    lambdas: Vec<f64>,
}

impl JointPenaltyBundle {
    /// Build a bundle, validating the per-penalty `log λ` count and dimension
    /// agreement against `total_compiled`.
    pub fn new(
        specs: std::sync::Arc<Vec<JointPenaltySpec>>,
        log_lambdas: Vec<f64>,
        total_compiled: usize,
    ) -> Result<Self, String> {
        let roots = specs
            .iter()
            .enumerate()
            .map(|(index, spec)| {
                spec.validated_root()
                    .map_err(|error| format!("joint penalty {index}: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_validated_geometry(
            specs,
            std::sync::Arc::new(roots),
            log_lambdas,
            total_compiled,
        )
    }

    /// Build a rho-specific bundle from already-validated invariant geometry.
    ///
    /// `specs` and `roots` are constructed together by the label-layout
    /// compiler and retained across every outer evaluation. Only
    /// `log_lambdas` changes here. Shape and finiteness are still checked at
    /// this boundary; the expensive spectral identity was certified when the
    /// roots were created.
    pub fn from_validated_geometry(
        specs: std::sync::Arc<Vec<JointPenaltySpec>>,
        roots: std::sync::Arc<Vec<Array2<f64>>>,
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
        if roots.len() != specs.len() {
            return Err(format!(
                "joint penalty bundle: {} specs vs {} cached roots",
                specs.len(),
                roots.len(),
            ));
        }
        let mut lambdas = Vec::with_capacity(log_lambdas.len());
        for (i, ((spec, root), &log_lambda)) in specs
            .iter()
            .zip(roots.iter())
            .zip(log_lambdas.iter())
            .enumerate()
        {
            if spec.dim() != total_compiled {
                return Err(format!(
                    "joint penalty {i}: dim {} != total_compiled {}",
                    spec.dim(),
                    total_compiled,
                ));
            }
            if root.dim() != (spec.pseudo_rank(), total_compiled) {
                return Err(format!(
                    "joint penalty {i}: cached root shape {}x{} != rank-by-dimension {}x{}",
                    root.nrows(),
                    root.ncols(),
                    spec.pseudo_rank(),
                    total_compiled,
                ));
            }
            if let Some(((row, column), &value)) =
                root.indexed_iter().find(|(_, value)| !value.is_finite())
            {
                return Err(format!(
                    "joint penalty {i}: cached root has non-finite entry at ({row},{column}): {value}"
                ));
            }
            lambdas.push(
                crate::checked_exp_log_strength(log_lambda)
                    .map_err(|error| format!("joint penalty {i} current log-precision: {error}"))?,
            );
        }
        Ok(Self {
            specs,
            roots,
            log_lambdas,
            lambdas,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    #[inline]
    pub fn specs(&self) -> &[JointPenaltySpec] {
        self.specs.as_slice()
    }

    #[inline]
    pub fn roots(&self) -> &[Array2<f64>] {
        self.roots.as_slice()
    }

    #[inline]
    pub fn log_lambdas(&self) -> &[f64] {
        self.log_lambdas.as_slice()
    }

    #[inline]
    pub fn lambdas(&self) -> &[f64] {
        self.lambdas.as_slice()
    }

    /// Total joint-penalty contribution to the objective,
    /// `½ Σ_j exp(ρ_j)·‖R_j β‖²` on each component's structural root
    /// ([`JointPenaltySpec::validated_root`], `S_j = R_jᵀR_j`). The root form
    /// carries no `λ`-amplified null-space rounding (#2954,
    /// [`structural_penalty_root`]).
    pub fn quadratic(&self, beta: ArrayView1<'_, f64>) -> f64 {
        let mut total = 0.0;
        for (root, &lam) in self.roots.iter().zip(self.lambdas.iter()) {
            let root_beta = root.dot(&beta);
            total += 0.5 * lam * root_beta.dot(&root_beta);
        }
        total
    }

    /// Accumulate `Σ_j exp(ρ_j)·R_jᵀ(R_j v)` into `out` (additive): the
    /// gradient and Hessian action of [`Self::quadratic`].
    pub fn add_apply_into(&self, vector: ArrayView1<'_, f64>, out: &mut ndarray::Array1<f64>) {
        assert_eq!(out.len(), vector.len());
        for (root, &lam) in self.roots.iter().zip(self.lambdas.iter()) {
            let root_vector = root.dot(&vector);
            out.scaled_add(lam, &root.t().dot(&root_vector));
        }
    }

    /// Accumulate `Σ_j exp(ρ_j)·diag(R_jᵀR_j)` into `diag` (additive).
    pub fn add_diag(&self, diag: &mut ndarray::Array1<f64>) {
        for (root, &lam) in self.roots.iter().zip(self.lambdas.iter()) {
            for (i, column) in root.columns().into_iter().enumerate() {
                diag[i] += lam * column.dot(&column);
            }
        }
    }

    /// Accumulate `Σ_j exp(ρ_j)·R_jᵀR_j` into the full `matrix` (additive): the
    /// Hessian of [`Self::quadratic`].
    pub fn add_to_matrix(&self, matrix: &mut Array2<f64>) {
        assert_eq!(matrix.nrows(), matrix.ncols());
        for (root, &lam) in self.roots.iter().zip(self.lambdas.iter()) {
            matrix.scaled_add(lam, &root.t().dot(root));
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
            group: None,
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

    /// #2954: the bundle's quadratic is formed on each component's structural
    /// root, so it matches the structural value `½λ(0.25 + 1.0)` of
    /// [`cross_block_spec`] within the root form's own forward error, and its
    /// null directions contribute nothing at any `λ`.
    #[test]
    fn the_bundle_quadratic_is_the_root_form_at_any_strength_2954() {
        let specs = std::sync::Arc::new(vec![cross_block_spec()]);
        let beta: Array1<f64> = array![0.5, -0.25, 1.0, 0.75];
        // (1, 0, 1, 0) and (0, 1, 0, 1) span ker(S).
        let null: Array1<f64> = array![1.0, 0.0, 1.0, 0.0];
        for log_lambda in [0.0, 30.0] {
            let bundle = JointPenaltyBundle::new(specs.clone(), vec![log_lambda], 4)
                .expect("a valid bundle");
            let lambda = bundle.lambdas()[0];
            let value = bundle.quadratic(beta.view());
            let expected = 0.5 * lambda * 1.25;
            let band = gam_linalg::roundoff::accumulation_growth(16) * 2.0 * expected;
            assert!(
                (value - expected).abs() <= band,
                "λ={lambda:e}: {value:e} against {expected:e}, band {band:.3e}"
            );
            let null_value = bundle.quadratic(null.view());
            let root_band = lambda
                * (4.0 * f64::EPSILON * 2.0 * null.iter().map(|v| v.abs()).sum::<f64>()).powi(2);
            assert!(
                null_value.abs() <= root_band,
                "λ={lambda:e}: a null direction carries {null_value:e}, above {root_band:.3e}"
            );
        }
    }

    /// #2954: the declared nullity and the spectrum each remove directions from a
    /// penalty's root, and neither refuses the other. A formed penalty stores its
    /// structural zeros with its construction's rounding, which can sit above the
    /// eigensolver's Weyl band (the link wiggle's order-3 I-spline roughness,
    /// gam#2921: `1.687e-11` against `1.370e-11`), so only the declaration removes
    /// them. A declared range direction the spectrum cannot resolve is removed by
    /// the spectrum. A declaration that removes a resolved direction is a
    /// structural statement the root cannot audit without the construction's
    /// formation band.
    #[test]
    fn a_declared_nullity_and_the_spectrum_each_remove_directions_2954() {
        let exact = cross_block_spec().matrix;
        let rows = |matrix: &Array2<f64>, declared: Option<usize>| {
            structural_penalty_root(matrix, declared, 0.0)
                .expect("a square penalty with a nullity at most its dimension roots")
                .nrows()
        };
        assert_eq!(rows(&exact, Some(2)), 2);
        assert_eq!(rows(&exact, None), 2);
        assert_eq!(
            rows(&exact, Some(1)),
            2,
            "the spectrum removes an unresolved zero"
        );
        assert_eq!(
            rows(&exact, Some(3)),
            1,
            "the declaration removes a resolved direction"
        );
        // The same penalty with a formation error of eight Weyl bands left along
        // its structural null direction `(1, 0, 1, 0)/√2`.
        let weyl = 4.0 * f64::EPSILON * 2.0;
        let null = array![1.0, 0.0, 1.0, 0.0].mapv(|value: f64| value / 2.0_f64.sqrt());
        let mut formed = exact.clone();
        for i in 0..4 {
            for j in 0..4 {
                formed[[i, j]] += 8.0 * weyl * null[i] * null[j];
            }
        }
        assert_eq!(
            rows(&formed, None),
            3,
            "the spectrum resolves the formation error"
        );
        let root = structural_penalty_root(&formed, Some(2), 0.0).expect("declared nullity 2");
        assert_eq!(root.nrows(), 2);
        let along_null = root.dot(&null).mapv(|value| value * value).sum();
        assert!(
            along_null <= weyl,
            "the declared root carries {along_null:e} along the structural null direction"
        );
        // A stored null eigenvalue of `0.89·u·‖S‖` (the 2923 penalty's) is a
        // structural zero, inside the Weyl band, and is not counted.
        let stored_null = 0.89 * 0.5 * f64::EPSILON * 2.0;
        assert_eq!(
            gam_linalg::roundoff::resolved_eigenvalue_count(&[stored_null, 2.0, 2.0, 2.0], 0.0),
            3
        );
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
            group: None,
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
            group: None,
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
            group: None,
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
    fn validate_rejects_initial_log_strength_outside_exact_domain() {
        let spec = JointPenaltySpec {
            label: None,
            matrix: Array2::zeros((2, 2)),
            initial_log_lambda: f64::NAN,
            nullspace_dim: 0,
            group: None,
        };
        assert!(matches!(
            spec.validate(),
            Err(JointPenaltyError::InitialLogStrengthOutOfDomain { .. })
        ));

        let mut finite_but_too_large = cross_block_spec();
        finite_but_too_large.initial_log_lambda = crate::LOG_STRENGTH_MAX + 1.0;
        assert!(matches!(
            finite_but_too_large.validate(),
            Err(JointPenaltyError::InitialLogStrengthOutOfDomain { .. })
        ));
    }

    #[test]
    fn bundle_construction_is_atomic_at_exact_log_strength_faces() {
        let specs = std::sync::Arc::new(vec![cross_block_spec(), cross_block_spec()]);
        let bundle = JointPenaltyBundle::new(
            specs.clone(),
            vec![crate::LOG_STRENGTH_MIN, crate::LOG_STRENGTH_MAX],
            4,
        )
        .expect("closed endpoints");
        for ((&actual, &log_strength), expected) in bundle
            .lambdas()
            .iter()
            .zip(bundle.log_lambdas())
            .zip([crate::LOG_STRENGTH_MIN.exp(), crate::LOG_STRENGTH_MAX.exp()])
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
            assert_eq!(actual.to_bits(), log_strength.exp().to_bits());
        }

        let error = JointPenaltyBundle::new(specs, vec![0.0, crate::LOG_STRENGTH_MAX + 1.0], 4)
            .expect_err("one invalid coordinate refuses the whole bundle");
        assert!(error.contains("joint penalty 1 current log-precision"));
    }

    #[test]
    fn bundle_rejects_dim_mismatch() {
        let spec = JointPenaltySpec {
            label: None,
            matrix: Array2::<f64>::eye(3),
            initial_log_lambda: 0.0,
            nullspace_dim: 0,
            group: None,
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
            group: None,
        };
        let err = JointPenaltyBundle::new(std::sync::Arc::new(vec![spec]), vec![], 2)
            .expect_err("count mismatch must reject");
        assert!(err.contains("specs vs"));
    }
}
