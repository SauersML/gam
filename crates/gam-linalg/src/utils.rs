use crate::LinalgError;
use crate::faer_ndarray::{
    FaerArrayView, FaerCholeskyFactor, FaerLinalgError, array2_to_matmut,
    factorize_symmetricwith_fallback,
};
use crate::faer_ndarray::{FaerCholesky, FaerEigh};
use crate::matrix::symmetrize_in_place;
use crate::pcg::{DotReduction, PcgCoreResult, PcgDiagnostics, PcgStop, pcg_core};
use faer::Side;
use ndarray::{
    Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView2, ArrayView3, Data, Dimension, s,
};

/// SplitMix64: deterministic 64-bit hash / streaming RNG step.
///
/// Canonical home for the implementation that previously lived as eight
/// module-local copies (gpu/kernels/hutchpp, terms/analytic_penalties,
/// solver/evidence, solver/reml/unified, inference/sample, inference/hmc,
/// families/cubic_cell_kernel, families/marginal_slope_shared). All call
/// sites used identical constants; this is the streaming form. For the
/// pure-hash flavour (single `u64 -> u64` with no externally retained
/// state) use [`splitmix64_hash`].
#[inline]
pub const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Pure-hash flavour of [`splitmix64`]: takes a single `u64` seed and
/// returns a mixed value without persisting state. Equivalent to
/// `{ let mut s = x; splitmix64(&mut s) }`.
#[inline]
pub const fn splitmix64_hash(x: u64) -> u64 {
    let mut state = x;
    splitmix64(&mut state)
}

/// Vertically concatenate 1D blocks into a single contiguous vector.
///
/// Blocks are copied in order into a freshly allocated `Array1` whose length
/// is the sum of the block lengths. Canonical home for the implementation that
/// previously lived as identical module-local copies in
/// `families/latent_survival.rs` and `families/survival_location_scale.rs`,
/// where it stacks per-segment offset vectors (entry / exit / derivative) into
/// one design offset.
pub fn stack_offsets(blocks: &[&Array1<f64>]) -> Array1<f64> {
    let total: usize = blocks.iter().map(|block| block.len()).sum();
    let mut out = Array1::<f64>::zeros(total);
    let mut row = 0usize;
    for block in blocks {
        let end = row + block.len();
        out.slice_mut(ndarray::s![row..end]).assign(block);
        row = end;
    }
    out
}

/// Rows per streaming chunk so each `chunk_rows × p` `f64` tile stays near an
/// 8 MiB working-set budget, clamped to `[256, 65_536]` and never exceeding
/// `n`. Canonical home for the row-chunk heuristic that previously lived as
/// byte-identical module-local copies in `solver/pirls` (sparse-native nnz
/// counting) and `terms/smooth` (linear-fit column conditioning). With `p == 0`
/// there is no per-row footprint, so the whole design is one chunk.
pub fn row_chunk_for_byte_budget(n: usize, p: usize) -> usize {
    const TARGET_BYTES: usize = 8 * 1024 * 1024;
    const MIN_ROWS: usize = 256;
    const MAX_ROWS: usize = 65_536;
    if p == 0 {
        return n.max(1);
    }
    (TARGET_BYTES / (p * 8))
        .clamp(MIN_ROWS, MAX_ROWS)
        .min(n.max(1))
}

/// Trace of the matrix product `tr(A·B) = Σ_{i,j} A[i,j]·B[j,i]`, computed
/// without forming the product. `A` is `m×k`, `B` is `k×m`. Canonical home for
/// the byte-identical double-loop reduction that lived as module-local copies
/// (`trace_product_dense` in `solver/gaussian_reml`, `trace_projected_cross` in
/// `solver/reml/unified`).
pub fn trace_of_product(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> f64 {
    let mut value = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            value += a[[i, j]] * b[[j, i]];
        }
    }
    value
}

/// Numerically stable softplus `log(1 + exp(x))`.
///
/// Uses the identity `softplus(x) = max(x, 0) + log1p(exp(-|x|))`, which
/// avoids both `exp` overflow for large positive `x` and `log(1)` cancellation
/// for large negative `x`. Previously duplicated as `stable_softplus` in
/// `terms/smooth.rs` and `families/gamlss.rs`.
#[inline]
pub fn stable_softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Numerically stable logistic `σ(x) = 1 / (1 + exp(-x))`.
///
/// Splits on the sign of `x` to keep both `exp` arguments non-positive and
/// avoid overflow:
///   σ(x) = 1 / (1 + exp(-x))   for x ≥ 0,
///   σ(x) = exp(x) / (1 + exp(x))   for x < 0.
///
/// Canonical home for the routine previously duplicated as `logistic` in
/// `terms/analytic_penalties.rs`, `sigmoid_stable` in `inference/hmc.rs`, and
/// `sigmoid_scalar` in `terms/sae/manifold/mod.rs` — all three were bit-identical.
#[inline]
pub fn stable_logistic(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Generic finiteness check for any `f64` ndarray view (1-D, 2-D, etc.).
#[inline]
pub fn array_is_finite<S, D>(values: &ArrayBase<S, D>) -> bool
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    values.iter().all(|v| v.is_finite())
}

/// Infinity norm of an `f64` iterator: `max |x|`. Centralises the
/// `iter().fold(0.0, |a, b| a.max(b.abs()))` idiom that appeared in
/// multiple call sites across `solver/pirls.rs`, `inference/predict_input.rs`,
/// and `terms/construction.rs`. Returns `0.0` for an empty iterator.
#[inline]
pub fn inf_norm<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    values.into_iter().fold(0.0_f64, |acc, x| acc.max(x.abs()))
}

const MAX_SOLVE_RETRIES: usize = 8;

/// A posteriori certificate for an unperturbed symmetric linear solve.
///
/// The reported backward error is the max-entry norm bound
///
/// `||A X - B||max / (n ||A||max ||X||max + ||B||max)`.
///
/// This denominator is the forward-error scale of a length-`n` dot product,
/// so the ratio is invariant to uniform rescaling of either side.  Products in
/// the denominator are evaluated in the log domain, which keeps the
/// certificate meaningful when `||A||max * ||X||max` would overflow even
/// though every matrix entry and the computed solution are representable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetricSolveCertificate {
    pub dimension: usize,
    pub matrix_max_abs: f64,
    pub solution_max_abs: f64,
    pub rhs_max_abs: f64,
    pub residual_max_abs: f64,
    pub max_norm_backward_error: f64,
    pub allowed_backward_error: f64,
}

/// Certified solution of an unperturbed symmetric vector system.
#[derive(Debug)]
pub struct CertifiedSymmetricSolution {
    solution: Array1<f64>,
    certificate: SymmetricSolveCertificate,
}

impl CertifiedSymmetricSolution {
    #[inline]
    pub fn solution(&self) -> &Array1<f64> {
        &self.solution
    }

    #[inline]
    pub fn certificate(&self) -> SymmetricSolveCertificate {
        self.certificate
    }

    #[inline]
    pub fn into_solution(self) -> Array1<f64> {
        self.solution
    }
}

/// Certified inverse of an unperturbed symmetric positive-definite matrix.
#[derive(Debug)]
pub struct CertifiedSpdInverse {
    inverse: Array2<f64>,
    certificate: SymmetricSolveCertificate,
}

/// Strict, unjittered Cholesky factor coupled to the exact matrix it
/// factorized, so every subsequent solve can be certified against that same
/// unperturbed matrix.
pub struct CertifiedSpdFactor<'a> {
    matrix: &'a Array2<f64>,
    matrix_max_abs: f64,
    factor: FaerCholeskyFactor,
    label: &'a str,
}

impl CertifiedSpdFactor<'_> {
    /// Solve one right-hand side and certify the residual against the original
    /// matrix retained by this factor.
    pub fn solve(
        &self,
        rhs: &Array1<f64>,
    ) -> Result<CertifiedSymmetricSolution, CertifiedSymmetricSolveError> {
        if rhs.len() != self.matrix.nrows() {
            return Err(CertifiedSymmetricSolveError::InvalidRhsShape {
                label: self.label.to_string(),
                expected: self.matrix.nrows(),
                actual: rhs.len(),
            });
        }
        let rhs_matrix = rhs.view().insert_axis(ndarray::Axis(1)).to_owned();
        let solution = self.factor.solve_mat(&rhs_matrix);
        let certificate = certify_symmetric_matrix_solution(
            &self.matrix,
            self.matrix_max_abs,
            &rhs_matrix,
            &solution,
            &self.label,
        )?;
        Ok(CertifiedSymmetricSolution {
            solution: solution.column(0).to_owned(),
            certificate,
        })
    }

    /// Solve multiple right-hand sides and return the solution plus its shared
    /// max-norm backward-error certificate.
    pub fn solve_matrix(
        &self,
        rhs: &Array2<f64>,
    ) -> Result<(Array2<f64>, SymmetricSolveCertificate), CertifiedSymmetricSolveError> {
        if rhs.nrows() != self.matrix.nrows() {
            return Err(CertifiedSymmetricSolveError::InvalidRhsShape {
                label: self.label.to_string(),
                expected: self.matrix.nrows(),
                actual: rhs.nrows(),
            });
        }
        let solution = self.factor.solve_mat(rhs);
        let certificate = certify_symmetric_matrix_solution(
            &self.matrix,
            self.matrix_max_abs,
            rhs,
            &solution,
            &self.label,
        )?;
        Ok((solution, certificate))
    }

    /// Invert the retained SPD matrix and certify `A A⁻¹ = I`.
    pub fn inverse(&self) -> Result<CertifiedSpdInverse, CertifiedSymmetricSolveError> {
        let rhs = Array2::<f64>::eye(self.matrix.nrows());
        let mut inverse = self.factor.solve_mat(&rhs);
        // Independent identity columns can differ by a few solve-roundoff bits;
        // project those bits back to the analytic symmetry, then recertify.
        symmetrize_in_place(&mut inverse);
        let certificate = certify_symmetric_matrix_solution(
            &self.matrix,
            self.matrix_max_abs,
            &rhs,
            &inverse,
            &self.label,
        )?;
        Ok(CertifiedSpdInverse {
            inverse,
            certificate,
        })
    }
}

impl CertifiedSpdInverse {
    #[inline]
    pub fn inverse(&self) -> &Array2<f64> {
        &self.inverse
    }

    #[inline]
    pub fn certificate(&self) -> SymmetricSolveCertificate {
        self.certificate
    }

    #[inline]
    pub fn into_inverse(self) -> Array2<f64> {
        self.inverse
    }
}

/// Why an unperturbed symmetric solve could not be certified.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CertifiedSymmetricSolveError {
    #[error("{label}: symmetric system must be non-empty and square, got {rows}x{cols}")]
    InvalidMatrixShape {
        label: String,
        rows: usize,
        cols: usize,
    },
    #[error("{label}: right-hand side must have {expected} rows, got {actual}")]
    InvalidRhsShape {
        label: String,
        expected: usize,
        actual: usize,
    },
    #[error("{label}: invalid residual-certificate inputs: {reason}")]
    InvalidCertificateInput { label: String, reason: String },
    #[error("{label}: matrix entry ({row}, {col}) is non-finite: {value:?}")]
    NonFiniteMatrix {
        label: String,
        row: usize,
        col: usize,
        value: f64,
    },
    #[error("{label}: right-hand side entry ({row}, {col}) is non-finite: {value:?}")]
    NonFiniteRhs {
        label: String,
        row: usize,
        col: usize,
        value: f64,
    },
    #[error(
        "{label}: matrix is not symmetric at ({row}, {col}): {lower:?} versus {upper:?} \
         (defect {defect:.3e} exceeds {tolerance:.3e})"
    )]
    NotSymmetric {
        label: String,
        row: usize,
        col: usize,
        lower: f64,
        upper: f64,
        defect: f64,
        tolerance: f64,
    },
    #[error("{label}: unperturbed symmetric factorization failed: {reason}")]
    Factorization { label: String, reason: String },
    #[error("{label}: matrix is not strictly positive definite: {reason}")]
    NotPositiveDefinite { label: String, reason: String },
    #[error("{label}: solution entry ({row}, {col}) is non-finite: {value:?}")]
    NonFiniteSolution {
        label: String,
        row: usize,
        col: usize,
        value: f64,
    },
    #[error("{label}: residual entry ({row}, {col}) is non-finite: {value:?}")]
    NonFiniteResidual {
        label: String,
        row: usize,
        col: usize,
        value: f64,
    },
    #[error(
        "{label}: unperturbed solve failed its backward-error certificate: \
         eta={backward_error:.3e} > {allowed:.3e} (max residual {residual_max_abs:.3e})"
    )]
    BackwardErrorTooLarge {
        label: String,
        backward_error: f64,
        allowed: f64,
        residual_max_abs: f64,
    },
}

const SYMMETRY_ULP_ALLOWANCE: f64 = 32.0;
const SOLVE_ROUNDOFF_OPS_PER_DIMENSION: f64 = 256.0;

#[inline]
fn positive_ulp(value: f64) -> f64 {
    debug_assert!(value.is_finite() && value >= 0.0);
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let next = f64::from_bits(value.to_bits() + 1);
    if next.is_finite() {
        next - value
    } else {
        value - f64::from_bits(value.to_bits() - 1)
    }
}

fn validate_symmetric_matrix(
    matrix: &Array2<f64>,
    label: &str,
) -> Result<f64, CertifiedSymmetricSolveError> {
    let (rows, cols) = matrix.dim();
    if rows == 0 || cols != rows {
        return Err(CertifiedSymmetricSolveError::InvalidMatrixShape {
            label: label.to_string(),
            rows,
            cols,
        });
    }
    let mut matrix_max_abs = 0.0_f64;
    for ((row, col), &value) in matrix.indexed_iter() {
        if !value.is_finite() {
            return Err(CertifiedSymmetricSolveError::NonFiniteMatrix {
                label: label.to_string(),
                row,
                col,
                value,
            });
        }
        matrix_max_abs = matrix_max_abs.max(value.abs());
    }
    for row in 0..rows {
        for col in 0..row {
            let lower = matrix[[row, col]];
            let upper = matrix[[col, row]];
            let defect = (lower - upper).abs();
            let pair_scale = lower.abs().max(upper.abs());
            let tolerance = SYMMETRY_ULP_ALLOWANCE * positive_ulp(pair_scale);
            if defect > tolerance {
                return Err(CertifiedSymmetricSolveError::NotSymmetric {
                    label: label.to_string(),
                    row,
                    col,
                    lower,
                    upper,
                    defect,
                    tolerance,
                });
            }
        }
    }
    Ok(matrix_max_abs)
}

#[inline]
fn max_abs_matrix(matrix: &Array2<f64>) -> f64 {
    matrix.iter().copied().map(f64::abs).fold(0.0_f64, f64::max)
}

fn max_norm_backward_error(
    dimension: usize,
    matrix_max_abs: f64,
    solution_max_abs: f64,
    rhs_max_abs: f64,
    residual_max_abs: f64,
) -> f64 {
    if residual_max_abs == 0.0 {
        return 0.0;
    }
    let product_log = if matrix_max_abs == 0.0 || solution_max_abs == 0.0 {
        f64::NEG_INFINITY
    } else {
        (dimension as f64).ln() + matrix_max_abs.ln() + solution_max_abs.ln()
    };
    let rhs_log = if rhs_max_abs == 0.0 {
        f64::NEG_INFINITY
    } else {
        rhs_max_abs.ln()
    };
    let largest = product_log.max(rhs_log);
    if largest == f64::NEG_INFINITY {
        return f64::INFINITY;
    }
    let denominator_log =
        largest + ((product_log - largest).exp() + (rhs_log - largest).exp()).ln();
    (residual_max_abs.ln() - denominator_log).exp()
}

#[inline]
fn solve_backward_error_allowance(dimension: usize) -> f64 {
    let roundoff = SOLVE_ROUNDOFF_OPS_PER_DIMENSION * dimension as f64 * f64::EPSILON;
    roundoff / (1.0 - roundoff)
}

fn certify_symmetric_matrix_solution(
    matrix: &Array2<f64>,
    matrix_max_abs: f64,
    rhs: &Array2<f64>,
    solution: &Array2<f64>,
    label: &str,
) -> Result<SymmetricSolveCertificate, CertifiedSymmetricSolveError> {
    let residual = matrix.dot(solution) - rhs;
    certify_linear_system_residual(
        matrix.nrows(),
        matrix_max_abs,
        rhs,
        solution,
        &residual,
        label,
    )
}

/// Certify a solve performed by an exact dense or sparse factorization from
/// its residual `A X - B` and the exact matrix max-entry norm.
///
/// This is the shared certification boundary for operator-backed systems that
/// cannot materialize `A` merely to call [`certified_symmetric_solve`].  It does
/// not establish symmetry or positive-definiteness; callers must obtain those
/// from their exact matrix representation and strict factorization.
pub fn certify_linear_system_residual(
    dimension: usize,
    matrix_max_abs: f64,
    rhs: &Array2<f64>,
    solution: &Array2<f64>,
    residual: &Array2<f64>,
    label: &str,
) -> Result<SymmetricSolveCertificate, CertifiedSymmetricSolveError> {
    if dimension == 0
        || !matrix_max_abs.is_finite()
        || matrix_max_abs < 0.0
        || rhs.nrows() != dimension
        || solution.dim() != rhs.dim()
        || residual.dim() != rhs.dim()
    {
        return Err(CertifiedSymmetricSolveError::InvalidCertificateInput {
            label: label.to_string(),
            reason: format!(
                "dimension={dimension}, matrix_max_abs={matrix_max_abs:?}, rhs={:?}, solution={:?}, residual={:?}",
                rhs.dim(),
                solution.dim(),
                residual.dim()
            ),
        });
    }
    for ((row, col), &value) in rhs.indexed_iter() {
        if !value.is_finite() {
            return Err(CertifiedSymmetricSolveError::NonFiniteRhs {
                label: label.to_string(),
                row,
                col,
                value,
            });
        }
    }
    for ((row, col), &value) in solution.indexed_iter() {
        if !value.is_finite() {
            return Err(CertifiedSymmetricSolveError::NonFiniteSolution {
                label: label.to_string(),
                row,
                col,
                value,
            });
        }
    }
    for ((row, col), &value) in residual.indexed_iter() {
        if !value.is_finite() {
            return Err(CertifiedSymmetricSolveError::NonFiniteResidual {
                label: label.to_string(),
                row,
                col,
                value,
            });
        }
    }
    let solution_max_abs = max_abs_matrix(solution);
    let rhs_max_abs = max_abs_matrix(rhs);
    let residual_max_abs = max_abs_matrix(residual);
    let max_norm_backward_error = max_norm_backward_error(
        dimension,
        matrix_max_abs,
        solution_max_abs,
        rhs_max_abs,
        residual_max_abs,
    );
    let allowed_backward_error = solve_backward_error_allowance(dimension);
    if !max_norm_backward_error.is_finite() || max_norm_backward_error > allowed_backward_error {
        return Err(CertifiedSymmetricSolveError::BackwardErrorTooLarge {
            label: label.to_string(),
            backward_error: max_norm_backward_error,
            allowed: allowed_backward_error,
            residual_max_abs,
        });
    }
    Ok(SymmetricSolveCertificate {
        dimension,
        matrix_max_abs,
        solution_max_abs,
        rhs_max_abs,
        residual_max_abs,
        max_norm_backward_error,
        allowed_backward_error,
    })
}

fn certified_symmetric_matrix_solve(
    matrix: &Array2<f64>,
    rhs: &Array2<f64>,
    label: &str,
) -> Result<(Array2<f64>, SymmetricSolveCertificate), CertifiedSymmetricSolveError> {
    let matrix_max_abs = validate_symmetric_matrix(matrix, label)?;
    if rhs.nrows() != matrix.nrows() {
        return Err(CertifiedSymmetricSolveError::InvalidRhsShape {
            label: label.to_string(),
            expected: matrix.nrows(),
            actual: rhs.nrows(),
        });
    }
    for ((row, col), &value) in rhs.indexed_iter() {
        if !value.is_finite() {
            return Err(CertifiedSymmetricSolveError::NonFiniteRhs {
                label: label.to_string(),
                row,
                col,
                value,
            });
        }
    }
    let factor = StableSolver::new(label)
        .factorize(matrix)
        .map_err(|error| CertifiedSymmetricSolveError::Factorization {
            label: label.to_string(),
            reason: error.to_string(),
        })?;
    let mut solution = rhs.clone();
    let mut solution_view = array2_to_matmut(&mut solution);
    factor.solve_in_place(solution_view.as_mut());
    let certificate =
        certify_symmetric_matrix_solution(matrix, matrix_max_abs, rhs, &solution, label)?;
    Ok((solution, certificate))
}

/// Solve `matrix * x = rhs` without adding a ridge, dropping a rank, or
/// changing the supplied estimand.  Singular and numerically unrepresentable
/// systems are errors; success carries an a posteriori backward-error proof.
pub fn certified_symmetric_solve(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    label: &str,
) -> Result<CertifiedSymmetricSolution, CertifiedSymmetricSolveError> {
    let mut rhs_matrix = Array2::<f64>::zeros((rhs.len(), 1));
    rhs_matrix.column_mut(0).assign(rhs);
    let (solution_matrix, certificate) =
        certified_symmetric_matrix_solve(matrix, &rhs_matrix, label)?;
    Ok(CertifiedSymmetricSolution {
        solution: solution_matrix.column(0).to_owned(),
        certificate,
    })
}

/// Strictly factor a finite symmetric positive-definite matrix without
/// diagonal jitter, spectral repair, or an indefinite LDLT/LBLT route.
pub fn certified_spd_factorize<'a>(
    matrix: &'a Array2<f64>,
    label: &'a str,
) -> Result<CertifiedSpdFactor<'a>, CertifiedSymmetricSolveError> {
    let matrix_max_abs = validate_symmetric_matrix(matrix, label)?;
    let factor = matrix.cholesky(Side::Lower).map_err(|error| {
        CertifiedSymmetricSolveError::NotPositiveDefinite {
            label: label.to_string(),
            reason: error.to_string(),
        }
    })?;
    Ok(CertifiedSpdFactor {
        matrix,
        matrix_max_abs,
        factor,
        label,
    })
}

/// Invert a symmetric positive-definite matrix without an additive
/// perturbation or spectral truncation.
///
/// Strict, unjittered Cholesky is the positive-definiteness certificate.  The
/// inverse is then accepted only when `A * A^-1 = I` also satisfies the
/// scale-aware backward-error certificate returned with it.  In particular,
/// this routine never routes through the repaired eigendecomposition API or an
/// LDLT/LBLT factorization that could bless an indefinite covariance.
pub fn certified_spd_inverse(
    matrix: &Array2<f64>,
    label: &str,
) -> Result<CertifiedSpdInverse, CertifiedSymmetricSolveError> {
    certified_spd_factorize(matrix, label)?.inverse()
}

#[derive(Default, Clone, Copy)]
pub struct KahanSum {
    sum: f64,
    c: f64,
}

impl KahanSum {
    #[inline]
    pub fn add(&mut self, value: f64) {
        let y = value - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    #[inline]
    pub fn sum(self) -> f64 {
        self.sum
    }
}

pub struct StableSolver<'a> {
    label: &'a str,
}

impl<'a> StableSolver<'a> {
    pub fn new(label: &'a str) -> Self {
        Self { label }
    }

    pub fn factorize(
        &self,
        matrix: &Array2<f64>,
    ) -> Result<crate::faer_ndarray::FaerSymmetricFactor, FaerLinalgError> {
        let view = FaerArrayView::new(matrix);
        factorize_symmetricwith_fallback(view.as_ref(), Side::Lower)
    }

    /// Generic factorize accepting any 2-D ndarray storage (owned or view).
    /// Useful for hot loops that solve a contiguous subblock of a hoisted
    /// workspace buffer without reallocating an owned `Array2`.
    pub fn factorize_any<S>(
        &self,
        matrix: &ArrayBase<S, ndarray::Ix2>,
    ) -> Result<crate::faer_ndarray::FaerSymmetricFactor, FaerLinalgError>
    where
        S: Data<Elem = f64>,
    {
        let view = FaerArrayView::new(matrix);
        factorize_symmetricwith_fallback(view.as_ref(), Side::Lower)
    }

    pub fn solvevectorwithridge_retries(
        &self,
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
        baseridge: f64,
    ) -> Option<Array1<f64>> {
        let p = matrix.nrows();
        if matrix.ncols() != p || rhs.len() != p {
            return None;
        }

        // Scale the ridge by the matrix's diagonal magnitude so it is
        // *rank-revealing* rather than absolute. A fixed `baseridge = 1e-10`
        // is meaningless for a Hessian whose largest diagonal is `O(1e8)`
        // (relative perturbation `1e-18` — well below f64 round-off) and
        // simultaneously over-regularises a diagonal of `O(1e-5)`. Anchoring
        // the ridge to `max_abs_diag(H)` makes the relative regularisation
        // strength independent of how the family scales its likelihood, so
        // null directions (eigenvalues < ridge) get treated consistently
        // across blocks. Without this, the joint-Newton solver returns
        // proposals with `|prop|∞ ≈ |g|/σ_min(H) = O(1e5–1e12)` because the
        // absolute `1e-10` ridge cannot reach the smallest eigenvalue of an
        // O(1e-5)-scale block while the largest block has `σ_max = 1e8`.
        let diag_scale = max_abs_diag(matrix);
        for retry in 0..MAX_SOLVE_RETRIES {
            let ridge = if baseridge > 0.0 {
                baseridge * diag_scale * 10f64.powi(retry as i32)
            } else {
                0.0
            };
            let h = addridge(matrix, ridge);
            let factor = match self.factorize(&h) {
                Ok(f) => f,
                Err(_) => continue,
            };
            let mut out = rhs.clone();
            let mut out_mat = crate::faer_ndarray::array1_to_col_matmut(&mut out);
            factor.solve_in_place(out_mat.as_mut());
            if out.iter().all(|v| v.is_finite()) {
                return Some(out);
            }
        }
        None
    }

    /// Solve `matrix · δ = rhs` with a rank-revealing fallback for the
    /// case where `matrix` has a near-null subspace aligned with `rhs`.
    ///
    /// First attempts the regularised Cholesky path
    /// (`solvevectorwithridge_retries`). If the produced δ satisfies the
    /// linear equation well (`‖matrix·δ − rhs‖∞ / (1 + ‖rhs‖∞) < rel_tol`),
    /// returns it. Otherwise the matrix has a real null subspace and the
    /// Tikhonov-regularised Newton step leaves a residual of magnitude
    /// ≈ ‖rhs_null‖ — the joint-Newton convergence test then fails
    /// (`linearized_rel ≈ 1`) and the seed is rejected.
    ///
    /// In that case we fall back to the truncated-eigendecomposition
    /// pseudoinverse:
    ///
    /// ```text
    /// δ = Σ_k (uₖᵀ rhs / λₖ) · uₖ      for k with |λₖ| > cutoff
    /// ```
    ///
    /// where `(λₖ, uₖ)` are the eigenpairs of `matrix` (assumed symmetric).
    /// Components in `null(matrix)` (i.e. |λₖ| ≤ cutoff) are *excluded* from
    /// the sum. This is the unique minimum-norm least-squares solution to
    /// `matrix · δ ≈ rhs`. For components of `rhs` in `range(matrix)`, δ
    /// solves the equation exactly; for components in `null(matrix)`, δ has
    /// zero contribution (no spurious huge step) and the joint-Newton's
    /// constrained-stationary certificate sees a *correctly small*
    /// projected residual.
    ///
    /// The cutoff is `rank_tol × max(|λ|)`, the standard rank-revealing
    /// threshold. For p ≲ a few hundred (joint Newton at large scale
    /// has p = 33) the eigendecomposition is sub-millisecond and saves
    /// the entire outer optimisation from rejecting ill-conditioned ρ.
    pub fn solve_with_pseudoinverse_fallback(
        &self,
        matrix: &Array2<f64>,
        rhs: &Array1<f64>,
        baseridge: f64,
        rel_tol: f64,
        rank_tol: f64,
    ) -> Option<Array1<f64>> {
        use crate::faer_ndarray::FaerEigh;
        use faer::Side;

        let p = matrix.nrows();
        if matrix.ncols() != p || rhs.len() != p {
            return None;
        }

        // First try the regularised Cholesky path.
        let delta = self.solvevectorwithridge_retries(matrix, rhs, baseridge)?;

        // Compute the linear residual ‖matrix·δ − rhs‖∞ / (1 + ‖rhs‖∞)
        // — the same quantity the joint-Newton convergence test reads off as
        // `linearized_next_kkt_inf` / (1 + `old_kkt_inf`).
        let matrix_delta = matrix.dot(&delta);
        let residual_inf = matrix_delta
            .iter()
            .zip(rhs.iter())
            .map(|(h, r)| (h - r).abs())
            .fold(0.0_f64, f64::max);
        let rhs_inf = rhs.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let rel = residual_inf / (1.0 + rhs_inf);

        if rel.is_finite() && rel < rel_tol {
            return Some(delta);
        }

        // Rank-deficient. Use truncated eigendecomposition pseudoinverse.
        let (eigvals, eigvecs) = matrix.eigh(Side::Lower).ok()?;
        let max_abs_eig = eigvals.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if !max_abs_eig.is_finite() || max_abs_eig <= 0.0 {
            return Some(delta);
        }
        let cutoff = rank_tol * max_abs_eig;

        let mut pseudo = Array1::<f64>::zeros(p);
        let mut excluded = 0usize;
        for k in 0..p {
            let lam = eigvals[k];
            if !lam.is_finite() || lam.abs() <= cutoff {
                excluded += 1;
                continue;
            }
            let u_k = eigvecs.column(k);
            let proj = u_k.iter().zip(rhs.iter()).map(|(u, r)| u * r).sum::<f64>();
            let scale = proj / lam;
            for i in 0..p {
                pseudo[i] += scale * u_k[i];
            }
        }

        if !pseudo.iter().all(|v| v.is_finite()) {
            return Some(delta);
        }

        log::debug!(
            "[{}] pseudoinverse fallback engaged: rel = {:.3e} > rel_tol = {:.3e}, \
             excluded {} of {} eigenvalues below cutoff = {:.3e} × max |λ| = {:.3e}",
            self.label,
            rel,
            rel_tol,
            excluded,
            p,
            rank_tol,
            max_abs_eig,
        );

        Some(pseudo)
    }
}

pub fn max_abs_diag(matrix: &Array2<f64>) -> f64 {
    matrix
        .diag()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0)
}

pub fn row_mismatch_message(
    y_len: usize,
    w_len: usize,
    x_rows: usize,
    offset_len: usize,
) -> Option<String> {
    if y_len == w_len && y_len == x_rows && y_len == offset_len {
        None
    } else {
        Some(format!(
            "Row mismatch: y={}, w={}, X.rows={}, offset={}",
            y_len, w_len, x_rows, offset_len
        ))
    }
}

pub fn predict_gam_dimension_mismatch_message(
    x_rows: usize,
    x_cols: usize,
    beta_len: usize,
    offset_len: usize,
) -> Option<String> {
    if x_cols != beta_len {
        return Some(format!(
            "predict_gam dimension mismatch: X has {} columns but beta has length {}",
            x_cols, beta_len
        ));
    }
    if x_rows != offset_len {
        return Some(format!(
            "predict_gam dimension mismatch: X has {} rows but offset has length {}",
            x_rows, offset_len
        ));
    }
    None::<String>
}

pub fn add_relative_diag_ridge(matrix: &mut Array2<f64>, scale: f64, floor: f64) -> f64 {
    let ridge = scale
        * matrix
            .diag()
            .iter()
            .map(|&value| value.abs())
            .fold(0.0, f64::max)
            .max(floor);
    for idx in 0..matrix.nrows() {
        matrix[[idx, idx]] += ridge;
    }
    ridge
}

pub fn boundary_hit_indices(
    values: ArrayView1<'_, f64>,
    bound: f64,
    tolerance: f64,
) -> (Vec<usize>, Vec<usize>) {
    let at_lower = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value <= -bound + tolerance).then_some(idx))
        .collect();
    let at_upper = values
        .iter()
        .enumerate()
        .filter_map(|(idx, &value)| (value >= bound - tolerance).then_some(idx))
        .collect();
    (at_lower, at_upper)
}

/// SPD-only spectrum condition number: λ_max / λ_min on the principal
/// (positive-eigenvalue) spectrum.
///
/// **Invariant:** caller must have already established the matrix is
/// positive definite. For indefinite matrices λ_min may be negative or
/// zero and the ratio max/min becomes meaningless (it can be negative or
/// infinite even when the matrix is well-scaled). When the spectrum sign is
/// unknown, inspect inertia directly via [`symmetric_extremes`].
pub fn symmetric_spectrum_condition_number(matrix: &Array2<f64>) -> f64 {
    matrix
        .eigh(Side::Lower)
        .ok()
        .map(|(evals, _)| {
            let min = evals
                .iter()
                .fold(f64::INFINITY, |acc, &value| acc.min(value));
            let max = evals
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &value| acc.max(value));
            max / min.max(1e-12)
        })
        .unwrap_or(f64::NAN)
}

pub fn addridge(matrix: &Array2<f64>, ridge: f64) -> Array2<f64> {
    if ridge <= 0.0 {
        return matrix.clone();
    }
    let mut regularized = matrix.clone();
    let n = regularized.nrows();
    for i in 0..n {
        regularized[[i, i]] += ridge;
    }
    regularized
}

pub fn boundary_hit_step_fraction(
    slack: f64,
    directional_slack_change: f64,
    current_step_limit: f64,
) -> Option<f64> {
    if !slack.is_finite()
        || !directional_slack_change.is_finite()
        || !current_step_limit.is_finite()
        || current_step_limit <= 0.0
    {
        return None;
    }

    let scale = slack
        .abs()
        .max(directional_slack_change.abs())
        .max(current_step_limit.abs())
        .max(1.0);
    let directional_tol = (64.0 * f64::EPSILON * scale).max(1e-14);
    if directional_slack_change >= -directional_tol {
        return None;
    }

    let step = (slack / -directional_slack_change).max(0.0);
    if step.is_finite() && step < current_step_limit {
        return Some(step);
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgSolveInfo {
    pub iterations: usize,
    pub converged: bool,
    pub relative_residual_norm: f64,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub residual_reduction: f64,
    pub condition_estimate: Option<f64>,
}

/// Ritz-based condition-number estimate from a PCG run's per-iteration trace.
///
/// Builds the CG Lanczos tridiagonal for the preconditioned operator. For SPD
/// CG, T has diagonal `1/a_i + b_{i-1}/a_{i-1}` and off-diagonal
/// `sqrt(b_i)/a_i`. Its eigenvalues are the Ritz estimates of the
/// preconditioned operator's spectrum; `cond ≈ λ_max(T) / λ_min(T)`.
///
/// (Gershgorin disc bounds were tried previously: they are guaranteed
/// *enclosures*, not estimates — systematically pessimistic, frequently
/// producing a negative lower bound even for SPD T and collapsing the estimate
/// to `None`. With `k ≤ 256` a direct symmetric eigensolve is microseconds and
/// yields the genuine Ritz values.)
fn pcg_condition_estimate(diagnostics: &PcgDiagnostics) -> Option<f64> {
    let alpha = &diagnostics.alpha;
    let beta = &diagnostics.beta;
    let k = alpha.len();
    if k == 0 || k > 256 {
        return None;
    }
    let mut t = ndarray::Array2::<f64>::zeros((k, k));
    for i in 0..k {
        let alpha_i = alpha[i];
        if !alpha_i.is_finite() || alpha_i <= 0.0 {
            return None;
        }
        let mut diag = 1.0 / alpha_i;
        if i > 0 {
            let beta_prev = beta.get(i - 1).copied()?;
            if !beta_prev.is_finite() || beta_prev < 0.0 {
                return None;
            }
            diag += beta_prev / alpha[i - 1];
        }
        t[[i, i]] = diag;
        if i + 1 < k {
            let beta_i = beta.get(i).copied().unwrap_or(0.0);
            if !beta_i.is_finite() || beta_i < 0.0 {
                return None;
            }
            let off = beta_i.sqrt() / alpha_i;
            t[[i, i + 1]] = off;
            t[[i + 1, i]] = off;
        }
    }
    let (evals, _) = t.eigh(Side::Lower).ok()?;
    let mut lower = f64::INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for &v in evals.iter() {
        if !v.is_finite() {
            return None;
        }
        if v < lower {
            lower = v;
        }
        if v > upper {
            upper = v;
        }
    }
    if lower > 0.0 && upper > 0.0 {
        Some(upper / lower)
    } else {
        None
    }
}

/// Assemble the public [`PcgSolveInfo`] from a finished [`pcg_core`] run.
fn pcg_solve_info(result: &PcgCoreResult) -> PcgSolveInfo {
    let rhs_norm = result.rhs_norm;
    let final_residual_norm = result.final_residual_norm;
    let initial = result
        .diagnostics
        .as_ref()
        .and_then(|d| d.residuals.first().copied())
        .unwrap_or(rhs_norm);
    // Report `‖r‖ / ‖rhs‖` — the textbook relative residual the
    // Eisenstat–Walker forcing term and the PCG stop condition both target.
    // When `‖rhs‖` is sub-unit, dividing by `max(‖rhs‖, 1)` understates the
    // true relative residual: e.g. `final = 5.3e-2`, `‖rhs‖ = 6.2e-2` is
    // reported as `5.3e-2` when the actual ratio is ~0.86 (one PCG iter
    // away from convergence, not 5% of the way). Match the stop criterion.
    let relative_residual_norm = if rhs_norm > 0.0 {
        final_residual_norm / rhs_norm
    } else {
        0.0
    };
    PcgSolveInfo {
        iterations: result.iterations,
        converged: result.stop == PcgStop::Converged,
        relative_residual_norm,
        initial_residual_norm: initial,
        final_residual_norm,
        residual_reduction: if initial > 0.0 {
            final_residual_norm / initial
        } else {
            0.0
        },
        condition_estimate: result.diagnostics.as_ref().and_then(pcg_condition_estimate),
    }
}

pub fn solve_spd_pcg_with_info<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<(Array1<f64>, PcgSolveInfo)>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    solve_spd_pcg_with_info_into(
        |v, out| {
            let applied = apply(v);
            if applied.len() == out.len() {
                out.assign(&applied);
            } else {
                out.fill(f64::NAN);
            }
        },
        rhs,
        preconditioner_diag,
        rel_tol,
        max_iter,
    )
}

pub fn solve_spd_pcg<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<Array1<f64>>
where
    F: Fn(&Array1<f64>) -> Array1<f64>,
{
    solve_spd_pcg_with_info(apply, rhs, preconditioner_diag, rel_tol, max_iter)
        .map(|(solution, _)| solution)
}

/// Write-into variant of `solve_spd_pcg_with_info` that takes an apply closure
/// of the form `Fn(&Array1<f64>, &mut Array1<f64>)` so the matvec can write into
/// a caller-owned buffer. This eliminates the per-iteration `Array1::<f64>`
/// allocation for the matvec result that the legacy closure-returning variant
/// forces. See commit 83369abb for the analogous penalty-vector elimination.
pub fn solve_spd_pcg_with_info_into<F>(
    apply: F,
    rhs: &Array1<f64>,
    preconditioner_diag: &Array1<f64>,
    rel_tol: f64,
    max_iter: usize,
) -> Option<(Array1<f64>, PcgSolveInfo)>
where
    F: Fn(&Array1<f64>, &mut Array1<f64>),
{
    let p = rhs.len();
    if p == 0 || preconditioner_diag.len() != p || max_iter == 0 {
        return None;
    }
    let mut x = Array1::<f64>::zeros(p);
    let result = pcg_core(
        apply,
        &rhs.view(),
        &preconditioner_diag.view(),
        rel_tol,
        max_iter,
        32,
        true,
        // Main SPD solve: strict serial reduction. This is the bit-identical-
        // across-threads / run-to-run contract the inexact-Newton callers and
        // the GPU-parity oracle depend on; it must never be relaxed here.
        DotReduction::Serial,
        &mut x.view_mut(),
    );
    if result.stop == PcgStop::Converged && x.iter().all(|v| v.is_finite()) {
        Some((x, pcg_solve_info(&result)))
    } else {
        if result.stop == PcgStop::BadPreconditioner {
            log::warn!(
                "SPD PCG rejected: preconditioner diagonal contained a non-positive or \
                 non-finite entry; caller should route to a direct factorization \
                 or indefinite Krylov path."
            );
        }
        None
    }
}

/// Weighted ridge (penalized least-squares) solve for a multi-output Gaussian
/// response. Forms the weighted normal equations `XᵀWX (+ λ·penalty) β = XᵀWY`
/// (row weights `W = diag(weights)`), factorizes the symmetric system via the
/// Cholesky-with-fallback path, solves for the coefficients `(p, d)`, and
/// returns `(coefficients, fitted = Xβ)`. Single source of truth shared by the
/// `gaussian_weighted_ridge` FFI shim and any core consumer.
pub fn gaussian_weighted_ridge(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    ridge_lambda: f64,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return Err("X cannot be empty".to_string());
    }
    if y.nrows() != n {
        return Err(format!(
            "X/Y row mismatch: X has {n} rows but Y has {} rows",
            y.nrows()
        ));
    }
    if y.ncols() == 0 {
        return Err("Y must have at least one column".to_string());
    }
    if weights.len() != n {
        return Err(format!(
            "weights length mismatch: expected {n}, got {}",
            weights.len()
        ));
    }
    if penalty.nrows() != p || penalty.ncols() != p {
        return Err(format!(
            "penalty shape mismatch: expected {p}x{p}, got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if !ridge_lambda.is_finite() || ridge_lambda < 0.0 {
        return Err(format!(
            "ridge_lambda must be finite and non-negative; got {ridge_lambda}"
        ));
    }
    if x.iter()
        .chain(y.iter())
        .chain(penalty.iter())
        .chain(weights.iter())
        .any(|value| !value.is_finite())
    {
        return Err("weighted ridge inputs must be finite".to_string());
    }
    if weights.iter().any(|value| *value < 0.0) {
        return Err("weights must be non-negative likelihood row weights".to_string());
    }

    let mut wx = x.to_owned();
    let mut wy = y.to_owned();
    for i in 0..n {
        let wi = weights[i];
        wx.row_mut(i).iter_mut().for_each(|value| *value *= wi);
        wy.row_mut(i).iter_mut().for_each(|value| *value *= wi);
    }
    let mut system = x.t().dot(&wx);
    if ridge_lambda > 0.0 {
        system += &(penalty.to_owned() * ridge_lambda);
    }
    let rhs = x.t().dot(&wy);
    let factor =
        factorize_symmetricwith_fallback(FaerArrayView::new(&system).as_ref(), Side::Lower)
            .map_err(|err| format!("weighted ridge factorization failed: {err}"))?;
    let mut coefficients = rhs;
    let mut coefficients_view = array2_to_matmut(&mut coefficients);
    factor.solve_in_place(coefficients_view.as_mut());
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err("weighted ridge solve produced non-finite coefficients".to_string());
    }
    let fitted = x.dot(&coefficients);
    Ok((coefficients, fitted))
}

/// Batched [`gaussian_weighted_ridge`]: solve one independent weighted-ridge fit
/// per leading-axis slice of the padded `(K, N_max, p)` design / `(K, N_max, d)`
/// response, honoring optional per-batch active `row_counts`. Runs the
/// per-batch solves in parallel and scatters results back into dense
/// `(K, p, d)` coefficients and `(K, N_max, d)` fitted arrays (padding rows
/// left zero).
pub fn gaussian_weighted_ridge_batch(
    x: ArrayView3<'_, f64>,
    y: ArrayView3<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView2<'_, f64>,
    ridge_lambda: f64,
    row_counts: Option<ArrayView1<'_, usize>>,
) -> Result<(Array3<f64>, Array3<f64>), String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let (batch, n_max, p) = x.dim();
    let (y_batch, y_n_max, d) = y.dim();
    if batch == 0 || n_max == 0 || p == 0 {
        return Err("batched X must have non-empty K, N, and coefficient dimensions".to_string());
    }
    if y_batch != batch || y_n_max != n_max {
        return Err(format!(
            "batched X/Y shape mismatch: X is ({batch}, {n_max}, {p}) but Y is ({y_batch}, {y_n_max}, {d})"
        ));
    }
    if d == 0 {
        return Err("batched Y must have at least one output column".to_string());
    }
    if weights.nrows() != batch || weights.ncols() != n_max {
        return Err(format!(
            "batched weights shape mismatch: expected ({batch}, {n_max}), got ({}, {})",
            weights.nrows(),
            weights.ncols()
        ));
    }
    if penalty.nrows() != p || penalty.ncols() != p {
        return Err(format!(
            "penalty shape mismatch: expected {p}x{p}, got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if !ridge_lambda.is_finite() || ridge_lambda < 0.0 {
        return Err(format!(
            "ridge_lambda must be finite and non-negative; got {ridge_lambda}"
        ));
    }
    if x.iter()
        .chain(y.iter())
        .chain(penalty.iter())
        .chain(weights.iter())
        .any(|value| !value.is_finite())
    {
        return Err("batched weighted ridge inputs must be finite".to_string());
    }
    if weights.iter().any(|value| *value < 0.0) {
        return Err("batched weights must be non-negative likelihood row weights".to_string());
    }

    let active_rows: Vec<usize> = match row_counts {
        Some(counts) => {
            if counts.len() != batch {
                return Err(format!(
                    "row_counts length mismatch: expected {batch}, got {}",
                    counts.len()
                ));
            }
            counts.to_vec()
        }
        None => vec![n_max; batch],
    };
    for (b, &n_rows) in active_rows.iter().enumerate() {
        if n_rows > n_max {
            return Err(format!(
                "row_counts[{b}]={n_rows} exceeds padded row count {n_max}"
            ));
        }
    }

    let results: Vec<Result<(usize, Array2<f64>, Array2<f64>), String>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let n_rows = active_rows[b];
            if n_rows == 0 {
                return Ok((
                    b,
                    Array2::<f64>::zeros((p, d)),
                    Array2::<f64>::zeros((0, d)),
                ));
            }
            gaussian_weighted_ridge(
                x.slice(s![b, 0..n_rows, ..]),
                y.slice(s![b, 0..n_rows, ..]),
                penalty,
                weights.slice(s![b, 0..n_rows]),
                ridge_lambda,
            )
            .map(|(coefficients, fitted)| (b, coefficients, fitted))
            .map_err(|err| format!("batched weighted ridge fit {b} failed: {err}"))
        })
        .collect();

    let mut coefficients = Array3::<f64>::zeros((batch, p, d));
    let mut fitted = Array3::<f64>::zeros((batch, n_max, d));
    for result in results {
        let (b, fit_coefficients, fit_fitted) = result?;
        coefficients
            .slice_mut(s![b, .., ..])
            .assign(&fit_coefficients);
        let n_rows = fit_fitted.nrows();
        if n_rows > 0 {
            fitted.slice_mut(s![b, 0..n_rows, ..]).assign(&fit_fitted);
        }
    }
    Ok((coefficients, fitted))
}

/// Rank and Moore–Penrose pseudoinverse of a symmetric PSD penalty matrix via
/// its eigendecomposition, keeping eigenpairs whose eigenvalue exceeds a
/// relative tolerance. Returns `(rank, pinv)`.
pub fn block_penalty_rank_and_pinv(
    penalty: &Array2<f64>,
) -> Result<(usize, Array2<f64>), LinalgError> {
    let (eigs, vecs) =
        penalty
            .to_owned()
            .eigh(Side::Lower)
            .map_err(|_| LinalgError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
    let max_abs = eigs.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let tol = (1.0e-10 * max_abs).max(1.0e-14);
    let mut rank = 0_usize;
    let mut scaled = Array2::<f64>::zeros(vecs.dim());
    for col in 0..eigs.len() {
        if eigs[col] > tol {
            rank += 1;
            for row in 0..vecs.nrows() {
                scaled[[row, col]] = vecs[[row, col]] / eigs[col];
            }
        }
    }
    Ok((rank, scaled.dot(&vecs.t())))
}

/// Invert a symmetric positive-definite matrix, escalating a relative diagonal
/// ridge until the Cholesky factorization succeeds (robust SPD inverse).
pub fn invert_spd_with_ridge(
    matrix: &Array2<f64>,
    ridge_rel: f64,
) -> Result<Array2<f64>, LinalgError> {
    let n = matrix.nrows();
    let eye = Array2::<f64>::eye(n);
    let scale = (0..n).map(|i| matrix[[i, i]].abs()).fold(1.0_f64, f64::max);
    let ridges = [0.0, ridge_rel, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4];
    for rel in ridges {
        let mut candidate = matrix.clone();
        if rel > 0.0 {
            for i in 0..n {
                candidate[[i, i]] += rel * scale;
            }
        }
        if let Ok(chol) = candidate.cholesky(Side::Lower) {
            return Ok(chol.solve_mat(&eye));
        }
    }
    Err(LinalgError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

/// Solve a symmetric (possibly indefinite/ill-conditioned) linear system via
/// eigendecomposition with a spectral floor: eigenvalues below the floor are
/// clamped (preserving sign) before inversion, stabilizing the solve.
pub fn solve_symmetric_vector_with_floor(
    matrix: &Array2<f64>,
    rhs: &Array1<f64>,
    ridge_rel: f64,
) -> Result<Array1<f64>, LinalgError> {
    let n = matrix.nrows();
    let mut sym = matrix.clone();
    symmetrize_in_place(&mut sym);
    let (eigs, vecs) = sym
        .eigh(Side::Lower)
        .map_err(|_| LinalgError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?;
    let max_eig = eigs.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let floor = (ridge_rel * max_eig.max(1.0)).max(1.0e-12);
    let projected = vecs.t().dot(rhs);
    let mut scaled = Array1::<f64>::zeros(n);
    for i in 0..n {
        let denom = if eigs[i].abs() >= floor {
            eigs[i]
        } else if eigs[i].is_sign_negative() {
            -floor
        } else {
            floor
        };
        scaled[i] = projected[i] / denom;
    }
    let out = vecs.dot(&scaled);
    if out.iter().all(|value| value.is_finite()) {
        Ok(out)
    } else {
        Err(LinalgError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })
    }
}

/// Solve a symmetric dense block system `H x = rhs` (single right-hand side)
/// via the Cholesky-with-fallback factorization, returning the solution vector.
/// `context` labels errors.
pub fn solve_dense_block_system(
    hessian: &Array2<f64>,
    rhs: &Array1<f64>,
    context: &str,
) -> Result<Array1<f64>, String> {
    certified_symmetric_solve(hessian, rhs, context)
        .map(CertifiedSymmetricSolution::into_solution)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod ridge_tests {
    use super::{gaussian_weighted_ridge, gaussian_weighted_ridge_batch};
    use ndarray::{Array2, Array3, ArrayView2, array, s};

    fn assert_close(lhs: ArrayView2<'_, f64>, rhs: ArrayView2<'_, f64>, tol: f64) {
        assert_eq!(lhs.dim(), rhs.dim());
        for ((i, j), value) in lhs.indexed_iter() {
            let diff = (*value - rhs[[i, j]]).abs();
            assert!(
                diff <= tol,
                "matrix mismatch at ({i}, {j}): lhs={}, rhs={}, diff={diff}",
                value,
                rhs[[i, j]]
            );
        }
    }

    #[test]
    fn weighted_ridge_batch_matches_single_fit_on_active_rows() {
        let x = Array3::from_shape_vec(
            (2, 3, 2),
            vec![1.0, 0.0, 1.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.0, 1.0, 9.0, 9.0],
        )
        .unwrap();
        let y = Array3::from_shape_vec((2, 3, 1), vec![1.0, 2.0, 1.5, 2.5, -0.5, 99.0]).unwrap();
        let weights = array![[1.0, 0.5, 2.0], [1.0, 3.0, 0.0]];
        let penalty = Array2::eye(2);
        let row_counts = array![3_usize, 2_usize];

        let (coefficients, fitted) = gaussian_weighted_ridge_batch(
            x.view(),
            y.view(),
            penalty.view(),
            weights.view(),
            0.25,
            Some(row_counts.view()),
        )
        .unwrap();

        for b in 0..2 {
            let n = row_counts[b];
            let (expected_coefficients, expected_fitted) = gaussian_weighted_ridge(
                x.slice(s![b, 0..n, ..]),
                y.slice(s![b, 0..n, ..]),
                penalty.view(),
                weights.slice(s![b, 0..n]),
                0.25,
            )
            .unwrap();
            assert_close(
                coefficients.slice(s![b, .., ..]),
                expected_coefficients.view(),
                1.0e-10,
            );
            assert_close(
                fitted.slice(s![b, 0..n, ..]),
                expected_fitted.view(),
                1.0e-10,
            );
        }
        assert_eq!(fitted[[1, 2, 0]], 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        boundary_hit_step_fraction, solve_spd_pcg, solve_spd_pcg_with_info,
        solve_spd_pcg_with_info_into, splitmix64, splitmix64_hash,
    };
    use ndarray::{Array1, array};

    /// Pin the canonical SplitMix64 stream to Vigna's reference sequence so the
    /// unification of the ~12 former module-local copies cannot drift seeds.
    #[test]
    fn splitmix64_matches_reference_sequence() {
        // Vigna's reference C `splitmix64` started from state 0.
        let mut state = 0u64;
        assert_eq!(splitmix64(&mut state), 0xE220A8397B1DCDAF);
        assert_eq!(splitmix64(&mut state), 0x6E789E6AA1B965F4);
        assert_eq!(splitmix64(&mut state), 0x06C45D188009454F);

        // The pure-hash flavour equals one stateful step seeded from `x`.
        for x in [0u64, 1, 42, 0x9E37_79B9_7F4A_7C15, u64::MAX] {
            let mut s = x;
            assert_eq!(splitmix64_hash(x), splitmix64(&mut s));
        }
    }

    /// Re-derive the literal three-line finalizer that every former copy
    /// inlined and confirm it is bit-identical to the canonical step. Guards
    /// against any future constant typo creeping into the single source.
    #[test]
    fn splitmix64_step_equals_inlined_finalizer() {
        for seed in [0u64, 7, 0xDEAD_BEEF, 0x0123_4567_89AB_CDEF, u64::MAX - 3] {
            let mut state = seed;
            let got = splitmix64(&mut state);

            let advanced = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = advanced;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let expect = z ^ (z >> 31);

            assert_eq!(got, expect);
            // The canonical step must have advanced state by exactly one G.
            assert_eq!(state, advanced);
        }
    }

    #[test]
    fn boundary_hit_step_fraction_ignores_near_tangential_direction() {
        let step = boundary_hit_step_fraction(1.0, -1e-16, 1.0);
        assert_eq!(step, None);
    }

    #[test]
    fn boundary_hit_step_fraction_returns_first_finite_hit() {
        let step = boundary_hit_step_fraction(0.25, -0.5, 1.0);
        assert_eq!(step, Some(0.5));
    }

    #[test]
    fn boundary_hit_step_fraction_rejects_non_finite_candidate() {
        let step = boundary_hit_step_fraction(1.0, f64::NEG_INFINITY, 1.0);
        assert_eq!(step, None);
    }

    #[test]
    fn solve_spd_pcg_matches_reference_solution() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        let x = solve_spd_pcg(|v| h.dot(v), &b, &m, 1e-10, 20).expect("pcg solve");
        assert!((x[0] - 0.0909090909).abs() < 1e-8);
        assert!((x[1] - 0.6363636363).abs() < 1e-8);
    }

    #[test]
    fn solve_spd_pcg_rejects_zero_iteration_budget() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        assert!(solve_spd_pcg_with_info(|v| h.dot(v), &b, &m, 1e-10, 0).is_none());
        assert!(solve_spd_pcg(|v| h.dot(v), &b, &m, 1e-10, 0).is_none());
    }

    #[test]
    fn matrix_free_qp_beta_matches_dense_reference_with_diagnostics() {
        // Small synthetic stand-in for the FLEX marginal-slope joint system:
        // a coupled SPD Hessian plus a penalty/ridge Jacobi preconditioner. The
        // matrix-free solve must return the same beta as the dense reference,
        // while surfacing bounded iteration/residual diagnostics for cycle-0
        // triage.
        let h = array![
            [12.0, 2.0, 0.5, 0.0],
            [2.0, 9.0, 1.25, 0.25],
            [0.5, 1.25, 7.0, 1.5],
            [0.0, 0.25, 1.5, 5.0],
        ];
        let rhs = array![1.0, -0.5, 2.0, 0.75];
        let precond = h.diag().to_owned();
        let factor = super::StableSolver::new("synthetic dense reference")
            .factorize(&h)
            .expect("dense SPD reference");
        let mut dense = rhs.clone();
        let mut dense_view = crate::faer_ndarray::array1_to_col_matmut(&mut dense);
        factor.solve_in_place(dense_view.as_mut());
        let (pcg, info) = solve_spd_pcg_with_info_into(
            |v, out| {
                let prod = h.dot(v);
                out.assign(&prod);
            },
            &rhs,
            &precond,
            1e-12,
            4 * rhs.len(),
        )
        .expect("matrix-free pcg");

        assert!(info.converged);
        assert!(info.iterations <= 4 * rhs.len());
        assert!(info.final_residual_norm < info.initial_residual_norm);
        assert!(info.residual_reduction < 1e-10);
        assert!(info.condition_estimate.is_some());
        for (reference, actual) in dense.iter().zip(pcg.iter()) {
            assert!(
                (reference - actual).abs() < 1e-10,
                "dense={reference} pcg={actual}"
            );
        }
    }

    #[test]
    fn solve_spd_pcg_with_info_into_rejects_zero_iteration_budget() {
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let m = Array1::from_vec(vec![4.0, 3.0]);
        assert!(
            solve_spd_pcg_with_info_into(
                |v, out| {
                    let prod = h.dot(v);
                    out.assign(&prod);
                },
                &b,
                &m,
                1e-10,
                0,
            )
            .is_none()
        );
    }
}

#[cfg(test)]
mod pure_fn_tests {
    use super::{
        addridge, inf_norm, max_abs_diag, predict_gam_dimension_mismatch_message,
        row_mismatch_message, stable_logistic, stable_softplus,
    };
    use ndarray::array;

    // -----------------------------------------------------------------------
    // stable_softplus: log(1 + exp(x))
    // -----------------------------------------------------------------------

    #[test]
    fn softplus_at_zero() {
        let got = stable_softplus(0.0);
        let expected = (1.0_f64 + 1.0_f64).ln();
        assert!((got - expected).abs() < 1e-14, "got={got}");
    }

    #[test]
    fn softplus_positive_large_approximates_x() {
        let x = 100.0_f64;
        let got = stable_softplus(x);
        assert!(
            (got - x).abs() < 1e-10,
            "softplus({x}) = {got}, expected ~{x}"
        );
    }

    #[test]
    fn softplus_negative_large_approximates_zero() {
        let x = -50.0_f64;
        let got = stable_softplus(x);
        assert!(got >= 0.0, "softplus must be non-negative, got {got}");
        assert!(got < 1e-10, "softplus({x}) = {got}, expected ~0");
    }

    #[test]
    fn softplus_matches_naive_formula_at_moderate_x() {
        for x in [-5.0_f64, -1.0, 0.5, 1.0, 5.0] {
            let got = stable_softplus(x);
            let expected = (1.0 + x.exp()).ln();
            assert!(
                (got - expected).abs() < 1e-12,
                "x={x}: got={got} expected={expected}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // stable_logistic: 1 / (1 + exp(-x))
    // -----------------------------------------------------------------------

    #[test]
    fn logistic_at_zero_is_half() {
        let got = stable_logistic(0.0);
        assert!((got - 0.5).abs() < 1e-15, "got={got}");
    }

    #[test]
    fn logistic_large_positive_approaches_one() {
        let got = stable_logistic(100.0);
        assert!((got - 1.0).abs() < 1e-10, "got={got}");
    }

    #[test]
    fn logistic_large_negative_approaches_zero() {
        let got = stable_logistic(-100.0);
        assert!(got >= 0.0 && got < 1e-10, "got={got}");
    }

    #[test]
    fn logistic_symmetry_around_zero() {
        for x in [0.5_f64, 1.0, 2.0, 5.0] {
            let pos = stable_logistic(x);
            let neg = stable_logistic(-x);
            assert!(
                (pos + neg - 1.0).abs() < 1e-15,
                "x={x}: pos={pos} neg={neg}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // inf_norm
    // -----------------------------------------------------------------------

    #[test]
    fn inf_norm_empty_is_zero() {
        assert_eq!(inf_norm(std::iter::empty()), 0.0);
    }

    #[test]
    fn inf_norm_all_positive() {
        assert_eq!(inf_norm([1.0, 2.0, 3.0]), 3.0);
    }

    #[test]
    fn inf_norm_mixed_signs() {
        assert_eq!(inf_norm([-5.0_f64, 2.0, -3.0]), 5.0);
    }

    // -----------------------------------------------------------------------
    // max_abs_diag
    // -----------------------------------------------------------------------

    #[test]
    fn max_abs_diag_floors_at_one() {
        let m = array![[0.1_f64, 0.0], [0.0, 0.2]];
        assert_eq!(max_abs_diag(&m), 1.0);
    }

    #[test]
    fn max_abs_diag_returns_largest_abs_diagonal() {
        let m = array![[3.0_f64, 99.0], [0.0, -7.0]];
        assert_eq!(max_abs_diag(&m), 7.0);
    }

    // -----------------------------------------------------------------------
    // addridge
    // -----------------------------------------------------------------------

    #[test]
    fn addridge_zero_ridge_clones_matrix() {
        let m = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let r = addridge(&m, 0.0);
        assert_eq!(r, m);
    }

    #[test]
    fn addridge_negative_ridge_clones_matrix() {
        let m = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let r = addridge(&m, -1.0);
        assert_eq!(r, m);
    }

    #[test]
    fn addridge_positive_adds_to_diagonal() {
        let m = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let r = addridge(&m, 0.5);
        assert_eq!(r[[0, 0]], 1.5);
        assert_eq!(r[[1, 1]], 2.5);
        assert_eq!(r[[0, 1]], 0.0);
    }

    // -----------------------------------------------------------------------
    // row_mismatch_message / predict_gam_dimension_mismatch_message
    // -----------------------------------------------------------------------

    #[test]
    fn row_mismatch_none_when_all_match() {
        assert_eq!(row_mismatch_message(5, 5, 5, 5), None);
    }

    #[test]
    fn row_mismatch_some_when_lengths_differ() {
        assert!(row_mismatch_message(5, 4, 5, 5).is_some());
    }

    #[test]
    fn predict_gam_mismatch_none_when_consistent() {
        assert_eq!(predict_gam_dimension_mismatch_message(10, 3, 3, 10), None);
    }

    #[test]
    fn predict_gam_mismatch_some_when_cols_differ() {
        assert!(predict_gam_dimension_mismatch_message(10, 3, 4, 10).is_some());
    }

    #[test]
    fn predict_gam_mismatch_some_when_rows_differ() {
        assert!(predict_gam_dimension_mismatch_message(10, 3, 3, 9).is_some());
    }
}
