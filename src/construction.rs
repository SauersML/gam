use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerEigh, FaerLinalgError, FaerSvd};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par, Side};
use ndarray::{Array1, Array2, ArrayViewMut2, Axis, s};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::ops::Range;

#[derive(Clone)]
pub enum PenaltyRepresentation {
    Dense(Array2<f64>),
    Banded {
        bands: Vec<Array1<f64>>,
        offsets: Vec<i32>,
    },
    Kronecker {
        left: Array2<f64>,
        right: Array2<f64>,
    },
}

impl PenaltyRepresentation {
    fn block_dimension(&self) -> usize {
        match self {
            PenaltyRepresentation::Dense(matrix) => matrix.nrows(),
            PenaltyRepresentation::Banded { bands, offsets } => {
                let mut dim = 0usize;
                for (band, &offset) in bands.iter().zip(offsets.iter()) {
                    let len = band.len();
                    let extent = if offset >= 0 {
                        len + offset as usize
                    } else {
                        len + (-offset) as usize
                    };
                    dim = dim.max(extent);
                }
                dim
            }
            PenaltyRepresentation::Kronecker { left, right } => left.nrows() * right.nrows(),
        }
    }

    fn to_block_dense(&self) -> Array2<f64> {
        match self {
            PenaltyRepresentation::Dense(matrix) => matrix.clone(),
            PenaltyRepresentation::Banded { bands, offsets } => {
                let dim = self.block_dimension();
                let mut dense = Array2::zeros((dim, dim));
                for (band, &offset) in bands.iter().zip(offsets.iter()) {
                    if offset >= 0 {
                        let off = offset as usize;
                        for (idx, &value) in band.iter().enumerate() {
                            dense[[idx, idx + off]] = value;
                        }
                    } else {
                        let off = (-offset) as usize;
                        for (idx, &value) in band.iter().enumerate() {
                            dense[[idx + off, idx]] = value;
                        }
                    }
                }
                dense
            }
            PenaltyRepresentation::Kronecker { left, right } => {
                let (l_rows, l_cols) = left.dim();
                let (r_rows, r_cols) = right.dim();
                let mut result = Array2::zeros((l_rows * r_rows, l_cols * r_cols));
                for i in 0..l_rows {
                    for j in 0..l_cols {
                        let scale = left[(i, j)];
                        if scale == 0.0 {
                            continue;
                        }
                        let mut block = result.slice_mut(s![
                            i * r_rows..(i + 1) * r_rows,
                            j * r_cols..(j + 1) * r_cols
                        ]);
                        block.assign(&(right * scale));
                    }
                }
                result
            }
        }
    }
}

#[derive(Clone)]
pub struct PenaltyMatrix {
    pub col_range: Range<usize>,
    pub representation: PenaltyRepresentation,
}

impl PenaltyMatrix {
    fn accumulate_into(&self, mut dest: ArrayViewMut2<'_, f64>, weight: f64) {
        if weight == 0.0 {
            return;
        }
        match &self.representation {
            PenaltyRepresentation::Dense(block) => {
                dest.scaled_add(weight, block);
            }
            PenaltyRepresentation::Banded { bands, offsets } => {
                for (band, &offset) in bands.iter().zip(offsets.iter()) {
                    if offset >= 0 {
                        let off = offset as usize;
                        for (idx, &value) in band.iter().enumerate() {
                            let entry = dest.get_mut((idx, idx + off)).expect("banded index");
                            *entry += weight * value;
                        }
                    } else {
                        let off = (-offset) as usize;
                        for (idx, &value) in band.iter().enumerate() {
                            let entry = dest.get_mut((idx + off, idx)).expect("banded index");
                            *entry += weight * value;
                        }
                    }
                }
            }
            PenaltyRepresentation::Kronecker { left, right } => {
                let (l_rows, l_cols) = left.dim();
                let (r_rows, r_cols) = right.dim();
                for i in 0..l_rows {
                    for j in 0..l_cols {
                        let scale = left[(i, j)] * weight;
                        if scale == 0.0 {
                            continue;
                        }
                        let mut block = dest.slice_mut(s![
                            i * r_rows..(i + 1) * r_rows,
                            j * r_cols..(j + 1) * r_cols
                        ]);
                        block.scaled_add(scale, right);
                    }
                }
            }
        }
    }

    pub fn to_dense(&self, total_dim: usize) -> Array2<f64> {
        let mut dense = Array2::<f64>::zeros((total_dim, total_dim));
        self.accumulate_into(
            dense.slice_mut(s![self.col_range.clone(), self.col_range.clone()]),
            1.0,
        );
        dense
    }

    pub fn block_dense(&self) -> Array2<f64> {
        self.representation.to_block_dense()
    }
}

fn max_abs_element(matrix: &Array2<f64>) -> f64 {
    matrix
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, |acc, &val| acc.max(val.abs()))
}

fn sanitize_symmetric(matrix: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = matrix.dim();
    debug_assert_eq!(rows, cols, "Matrix must be square for sanitization");

    let mut sanitized = matrix.clone();

    for i in 0..rows {
        let diag = sanitized[[i, i]];
        if !diag.is_finite() {
            sanitized[[i, i]] = 0.0;
        }
        for j in (i + 1)..cols {
            let mut upper = sanitized[[i, j]];
            let mut lower = sanitized[[j, i]];
            if !upper.is_finite() {
                upper = 0.0;
            }
            if !lower.is_finite() {
                lower = 0.0;
            }
            let avg = 0.5 * (upper + lower);
            sanitized[[i, j]] = avg;
            sanitized[[j, i]] = avg;
        }
    }

    let scale = max_abs_element(&sanitized);
    let tiny = (scale * 1e-14).max(1e-30);
    for val in sanitized.iter_mut() {
        if !val.is_finite() {
            *val = 0.0;
        } else if val.abs() < tiny {
            *val = 0.0;
        }
    }

    sanitized
}

fn array_to_faer(array: &Array2<f64>) -> Mat<f64> {
    let (rows, cols) = array.dim();
    Mat::from_fn(rows, cols, |i, j| array[[i, j]])
}

fn mat_to_array(mat: &Mat<f64>) -> Array2<f64> {
    Array2::from_shape_fn((mat.nrows(), mat.ncols()), |(i, j)| mat[(i, j)])
}

fn mat_max_abs_element(matrix: MatRef<'_, f64>) -> f64 {
    let (rows, cols) = matrix.shape();
    let mut max_val = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let val = matrix[(i, j)];
            if val.is_finite() {
                max_val = max_val.max(val.abs());
            }
        }
    }
    max_val
}

fn sanitize_symmetric_faer(matrix: &Mat<f64>) -> Mat<f64> {
    let (rows, cols) = matrix.as_ref().shape();
    debug_assert_eq!(rows, cols, "Matrix must be square for sanitization");

    let mut sanitized = matrix.clone();

    for i in 0..rows {
        let diag = sanitized[(i, i)];
        if !diag.is_finite() {
            sanitized[(i, i)] = 0.0;
        }
        for j in (i + 1)..cols {
            let mut upper = sanitized[(i, j)];
            let mut lower = sanitized[(j, i)];
            if !upper.is_finite() {
                upper = 0.0;
            }
            if !lower.is_finite() {
                lower = 0.0;
            }
            let avg = 0.5 * (upper + lower);
            sanitized[(i, j)] = avg;
            sanitized[(j, i)] = avg;
        }
    }

    let scale = mat_max_abs_element(sanitized.as_ref());
    let tiny = (scale * 1e-14).max(1e-30);
    for i in 0..rows {
        for j in 0..cols {
            let val = sanitized[(i, j)];
            if !val.is_finite() {
                sanitized[(i, j)] = 0.0;
            } else if val.abs() < tiny {
                sanitized[(i, j)] = 0.0;
            }
        }
    }

    sanitized
}

fn frobenius_norm_faer(matrix: &Mat<f64>) -> f64 {
    let (rows, cols) = matrix.as_ref().shape();
    let mut sum_sq = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let val = matrix[(i, j)];
            sum_sq += val * val;
        }
    }
    sum_sq.sqrt()
}

fn penalty_from_root_faer(root: &Mat<f64>) -> Mat<f64> {
    let cols = root.ncols();
    let mut full = Mat::<f64>::zeros(cols, cols);
    let root_ref = root.as_ref();
    let root_t = root_ref.transpose();
    matmul(
        full.as_mut(),
        Accum::Replace,
        root_t,
        root_ref,
        1.0,
        Par::Seq,
    );
    sanitize_symmetric_faer(&full)
}

fn robust_eigh_faer(
    matrix: &Mat<f64>,
    side: Side,
    context: &str,
) -> Result<(Vec<f64>, Mat<f64>), EstimationError> {
    let (rows, cols) = matrix.as_ref().shape();
    for i in 0..rows {
        for j in 0..cols {
            let val = matrix[(i, j)];
            if !val.is_finite() {
                let max_abs = mat_max_abs_element(matrix.as_ref());
                return Err(EstimationError::InvalidInput(format!(
                    "{} contains non-finite entries (max finite magnitude {:.3e})",
                    context, max_abs
                )));
            }
        }
    }

    let mut candidate = sanitize_symmetric_faer(matrix);
    let mut ridge = 0.0_f64;

    for attempt in 0..4 {
        match candidate.as_ref().self_adjoint_eigen(side) {
            Ok(eig) => {
                let diag = eig.S();
                let diag_len = diag.dim();
                let mut eigenvalues = Vec::with_capacity(diag_len);
                let mut scale = 0.0_f64;
                for idx in 0..diag_len {
                    let val = diag[idx];
                    if val.is_finite() {
                        scale = scale.max(val.abs());
                    }
                    eigenvalues.push(val);
                }
                let tolerance = if scale.is_finite() {
                    (scale * 1e-12).max(1e-12)
                } else {
                    1e-12
                };

                for val in eigenvalues.iter_mut() {
                    if !val.is_finite() {
                        *val = 0.0;
                        continue;
                    }
                    if val.abs() < tolerance {
                        *val = 0.0;
                    } else if *val < 0.0 {
                        if val.abs() <= tolerance * 10.0 {
                            *val = 0.0;
                        } else {
                            log::warn!(
                                "{} produced large negative eigenvalue {:.3e}; clamping for stability",
                                context,
                                *val
                            );
                            *val = 0.0;
                        }
                    }
                }

                let vectors_ref = eig.U();
                let mut eigenvectors = Mat::<f64>::zeros(vectors_ref.nrows(), vectors_ref.ncols());
                for i in 0..vectors_ref.nrows() {
                    for j in 0..vectors_ref.ncols() {
                        eigenvectors[(i, j)] = vectors_ref[(i, j)];
                    }
                }

                return Ok((eigenvalues, eigenvectors));
            }
            Err(err) => {
                if attempt == 3 {
                    return Err(EstimationError::EigendecompositionFailed(
                        FaerLinalgError::SelfAdjointEigen(err),
                    ));
                }

                let mut diag_scale = 0.0_f64;
                for idx in 0..candidate.nrows() {
                    let val = candidate[(idx, idx)];
                    if val.is_finite() {
                        diag_scale = diag_scale.max(val.abs());
                    }
                }
                let base = if diag_scale.is_finite() {
                    (diag_scale * 1e-8).max(1e-10)
                } else {
                    1e-8
                };

                ridge = if ridge == 0.0 { base } else { ridge * 10.0 };
                for idx in 0..candidate.nrows() {
                    candidate[(idx, idx)] += ridge;
                }

                log::warn!(
                    "{} eigendecomposition failed on attempt {}. Added ridge {:.3e} before retrying.",
                    context,
                    attempt + 1,
                    ridge
                );
            }
        }
    }

    unreachable!("robust_eigh_faer should return or error within 4 attempts")
}

fn transpose_owned(matrix: &Array2<f64>) -> Array2<f64> {
    let mut transposed = Array2::zeros((matrix.ncols(), matrix.nrows()));
    transposed.assign(&matrix.t());
    transposed
}

fn robust_eigh(
    matrix: &Array2<f64>,
    side: Side,
    context: &str,
) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
    if matrix.iter().any(|v| !v.is_finite()) {
        let max_abs = max_abs_element(matrix);
        return Err(EstimationError::InvalidInput(format!(
            "{} contains non-finite entries (max finite magnitude {:.3e})",
            context, max_abs
        )));
    }

    let mut candidate = sanitize_symmetric(matrix);
    let mut ridge = 0.0_f64;

    for attempt in 0..4 {
        match candidate.eigh(side) {
            Ok((mut eigenvalues, eigenvectors)) => {
                let scale = eigenvalues
                    .iter()
                    .filter(|v| v.is_finite())
                    .fold(0.0_f64, |acc, &val| acc.max(val.abs()));
                let tolerance = if scale.is_finite() {
                    (scale * 1e-12).max(1e-12)
                } else {
                    1e-12
                };

                for val in eigenvalues.iter_mut() {
                    if !val.is_finite() {
                        *val = 0.0;
                        continue;
                    }
                    if val.abs() < tolerance {
                        *val = 0.0;
                    } else if *val < 0.0 {
                        if val.abs() <= tolerance * 10.0 {
                            *val = 0.0;
                        } else {
                            log::warn!(
                                "{} produced large negative eigenvalue {:.3e}; clamping for stability",
                                context,
                                *val
                            );
                            *val = 0.0;
                        }
                    }
                }

                return Ok((eigenvalues, eigenvectors));
            }
            Err(err) => {
                if attempt == 3 {
                    return Err(EstimationError::EigendecompositionFailed(err));
                }

                let diag_scale = candidate
                    .diag()
                    .iter()
                    .filter(|v| v.is_finite())
                    .fold(0.0_f64, |acc, &val| acc.max(val.abs()));
                let base = if diag_scale.is_finite() {
                    (diag_scale * 1e-8).max(1e-10)
                } else {
                    1e-8
                };

                ridge = if ridge == 0.0 { base } else { ridge * 10.0 };
                for i in 0..candidate.nrows() {
                    candidate[[i, i]] += ridge;
                }

                log::warn!(
                    "{} eigendecomposition failed on attempt {}. Added ridge {:.3e} before retrying.",
                    context,
                    attempt + 1,
                    ridge
                );
            }
        }
    }

    unreachable!("robust_eigh should return or error within 4 attempts")
}

/// Computes weighted column means for functional ANOVA decomposition.
/// Returns the weighted means that would be subtracted by center_columns_in_place.
fn weighted_column_means(x: &Array2<f64>, w: &Array1<f64>) -> Array1<f64> {
    let denom = w.sum();
    if denom <= 0.0 {
        return Array1::zeros(x.ncols());
    }
    // Vectorized: means = (X^T w) / sum(w)
    x.t().dot(w) / denom
}

/// Centers the columns of a matrix using weighted means.
/// This enforces intercept orthogonality (sum-to-zero) for the columns it is applied to.
pub fn center_columns_in_place(x: &mut Array2<f64>, w: &Array1<f64>) {
    let means = weighted_column_means(x, w);
    // Subtract means from each column
    for j in 0..x.ncols() {
        let m = means[j];
        x.column_mut(j).mapv_inplace(|v| v - m);
    }
}

/// Computes the Kronecker product A ⊗ B for penalty matrix construction.
/// This is used to create tensor product penalties that enforce smoothness
/// in multiple dimensions for interaction terms.
pub fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    if a_rows == 0 || a_cols == 0 || b_rows == 0 || b_cols == 0 {
        return Array2::zeros((a_rows * b_rows, a_cols * b_cols));
    }
    let mut result = Array2::zeros((a_rows * b_rows, a_cols * b_cols));

    result
        .axis_chunks_iter_mut(Axis(0), b_rows)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row_block)| {
            let a_row = a.row(i);
            let col_chunks = row_block.axis_chunks_iter_mut(Axis(1), b_cols);
            for (j, mut block) in col_chunks.into_iter().enumerate() {
                let a_val = a_row[j];
                if a_val == 0.0 {
                    continue;
                }
                for (dest, &src) in block.iter_mut().zip(b.iter()) {
                    *dest = a_val * src;
                }
            }
        });

    result
}

/// Result of the stable reparameterization algorithm from Wood (2011) Appendix B
#[derive(Clone)]
pub struct ReparamResult {
    /// Transformed penalty matrix S
    pub s_transformed: Array2<f64>,
    /// Log-determinant of the penalty matrix (stable computation)
    pub log_det: f64,
    /// First derivatives of log-determinant w.r.t. log-smoothing parameters
    pub det1: Array1<f64>,
    /// Orthogonal transformation matrix Qs
    pub qs: Array2<f64>,
    /// Transformed penalty square roots rS (each is rank_k x p)
    pub rs_transformed: Vec<Array2<f64>>,
    /// Cached transposes of rS (each is p x rank_k) to avoid repeated transposes in hot paths
    pub rs_transposed: Vec<Array2<f64>>,
    /// Lambda-dependent penalty square root from s_transformed (rank x p matrix)
    /// This is used for applying the actual penalty in the least squares solve
    pub e_transformed: Array2<f64>,
    /// Truncated eigenvectors (p × m where m = p - structural_rank)
    /// These span the null space of the effective penalty and are needed for
    /// gradient correction: subtract tr(H⁻¹ P_⊥ S_k P_⊥) from the full trace.
    pub u_truncated: Array2<f64>,
}

/// Creates a lambda-independent balanced penalty root for stable rank detection
/// This follows mgcv's approach: scale each penalty to unit Frobenius norm, sum them,
/// and take the matrix square root. This balanced penalty is used ONLY for rank detection.
pub fn create_balanced_penalty_root(
    s_list: &[Array2<f64>],
    p: usize,
) -> Result<Array2<f64>, EstimationError> {
    if s_list.is_empty() {
        // No penalties case - return empty matrix with correct number of columns
        return Ok(Array2::zeros((0, p)));
    }

    // Validate penalty matrix dimensions
    for (idx, s) in s_list.iter().enumerate() {
        if s.nrows() != p || s.ncols() != p {
            return Err(EstimationError::LayoutError(format!(
                "Penalty matrix {idx} must be {p}×{p}, got {}×{}",
                s.nrows(),
                s.ncols()
            )));
        }
    }
    let mut s_balanced = Array2::zeros((p, p));

    // Scale each penalty to have unit Frobenius norm and sum them
    for s_k in s_list {
        let frob_norm = s_k.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if frob_norm > 1e-12 {
            // Scale to unit Frobenius norm and add to balanced sum
            s_balanced.scaled_add(1.0 / frob_norm, s_k);
        }
    }

    // Take the matrix square root of the balanced penalty
    let (eigenvalues, eigenvectors) =
        robust_eigh(&s_balanced, Side::Lower, "balanced penalty matrix")?;

    // Find the maximum eigenvalue to create a relative tolerance
    let max_eig = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));

    // Define a relative tolerance. Use an absolute fallback for zero matrices.
    let tolerance = if max_eig > 0.0 {
        max_eig * 1e-12
    } else {
        1e-12
    };

    let penalty_rank = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();

    if penalty_rank == 0 {
        return Ok(Array2::zeros((0, p)));
    }

    // Construct the balanced penalty square root
    let mut eb = Array2::zeros((p, penalty_rank));
    let mut col_idx = 0;
    for (i, &eigenval) in eigenvalues.iter().enumerate() {
        if eigenval > tolerance {
            let sqrt_eigenval = eigenval.sqrt();
            let eigenvec = eigenvectors.column(i);
            eb.column_mut(col_idx).assign(&(&eigenvec * sqrt_eigenval));
            col_idx += 1;
        }
    }

    // Return as rank x p matrix (matching mgcv's convention)
    Ok(eb.t().to_owned())
}

/// Computes penalty square roots from full penalty matrices using eigendecomposition
/// Returns "skinny" matrices of dimension rank_k x p where rank_k is the rank of each penalty
/// STANDARDIZED: All penalty roots use rank x p convention with S = R^T * R
pub fn compute_penalty_square_roots(
    s_list: &[Array2<f64>],
) -> Result<Vec<Array2<f64>>, EstimationError> {
    let mut rs_list = Vec::with_capacity(s_list.len());

    for s in s_list {
        let p = s.nrows();

        // Use eigendecomposition for symmetric positive semi-definite matrices
        let (eigenvalues, eigenvectors) = robust_eigh(s, Side::Lower, "penalty matrix")?;

        // Count positive eigenvalues to determine rank
        // Find the maximum eigenvalue to create a relative tolerance
        let max_eig = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));

        // Define a relative tolerance. Use an absolute fallback for zero matrices.
        let tolerance = if max_eig > 0.0 {
            max_eig * 1e-12
        } else {
            1e-12
        };

        let rank_k: usize = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();

        if rank_k == 0 {
            // Zero penalty matrix - return 0 x p matrix (STANDARDIZED: rank x p)
            rs_list.push(Array2::zeros((0, p)));
            continue;
        }

        // STANDARDIZED: Create rank x p square root matrix where S = rs^T * rs
        // Each row is sqrt(eigenvalue) * eigenvector^T
        let mut rs = Array2::zeros((rank_k, p));
        let mut row_idx = 0;

        for (i, &eigenval) in eigenvalues.iter().enumerate() {
            if eigenval > tolerance {
                let sqrt_eigenval = eigenval.sqrt();
                let eigenvec = eigenvectors.column(i);
                // Each row of rs is sqrt(eigenvalue) * eigenvector^T
                rs.row_mut(row_idx).assign(&(&eigenvec * sqrt_eigenval));
                row_idx += 1;
            }
        }

        rs_list.push(rs);
    }

    Ok(rs_list)
}

/// Helper to construct the summed, weighted penalty matrix S_lambda.
/// This version works with full-sized p × p penalty matrices from s_list.
pub fn construct_s_lambda(
    lambdas: &Array1<f64>,
    s_list: &[Array2<f64>],
    p: usize,
) -> Array2<f64> {
    let mut s_lambda = Array2::zeros((p, p));

    if s_list.is_empty() {
        return s_lambda;
    }

    // Validation: lambdas length must match number of penalty matrices
    if lambdas.len() != s_list.len() {
        panic!(
            "Lambda count mismatch: expected {} lambdas for {} penalty matrices, got {}",
            s_list.len(),
            s_list.len(),
            lambdas.len()
        );
    }

    // Simple weighted sum since all matrices are now p × p
    for (i, s_k) in s_list.iter().enumerate() {
        // Add weighted penalty matrix
        s_lambda.scaled_add(lambdas[i], s_k);
    }

    s_lambda
}

/// Lambda-independent reparameterization invariants derived from penalty structure.
#[derive(Clone)]
pub struct ReparamInvariant {
    qs_base: Array2<f64>,
    rs_transformed_base: Vec<Array2<f64>>,
    penalized_rank: usize,
    has_nonzero: bool,
}

/// Precompute the lambda-invariant reparameterization structure from penalty roots.
pub fn precompute_reparam_invariant(
    rs_list: &[Array2<f64>],
    p: usize,
) -> Result<ReparamInvariant, EstimationError> {
    use std::cmp::Ordering;

    let m = rs_list.len();

    if m == 0 {
        return Ok(ReparamInvariant {
            qs_base: Array2::eye(p),
            rs_transformed_base: Vec::new(),
            penalized_rank: 0,
            has_nonzero: false,
        });
    }

    let rs_faer: Vec<Mat<f64>> = rs_list.iter().map(array_to_faer).collect();
    let s_original_list: Vec<Mat<f64>> = rs_faer.iter().map(penalty_from_root_faer).collect();

    let mut s_balanced = Mat::<f64>::zeros(p, p);
    let mut has_nonzero = false;
    for s_k in &s_original_list {
        let frob_norm = frobenius_norm_faer(s_k);
        if frob_norm > 1e-12 {
            let scale = 1.0 / frob_norm;
            for i in 0..p {
                for j in 0..p {
                    s_balanced[(i, j)] += scale * s_k[(i, j)];
                }
            }
            has_nonzero = true;
        }
    }

    if !has_nonzero {
        return Ok(ReparamInvariant {
            qs_base: Array2::eye(p),
            rs_transformed_base: rs_list.to_vec(),
            penalized_rank: 0,
            has_nonzero: false,
        });
    }

    let (bal_eigenvalues, bal_eigenvectors) =
        robust_eigh_faer(&s_balanced, Side::Lower, "balanced penalty matrix")?;

    let mut order: Vec<usize> = (0..p).collect();
    order.sort_by(|&i, &j| {
        bal_eigenvalues[j]
            .partial_cmp(&bal_eigenvalues[i])
            .unwrap_or(Ordering::Equal)
            .then(i.cmp(&j))
    });

    let mut qs = Mat::<f64>::zeros(p, p);
    for (col_idx, &idx) in order.iter().enumerate() {
        for row in 0..p {
            qs[(row, col_idx)] = bal_eigenvectors[(row, idx)];
        }
    }

    let mut bal_eigenvalues_ordered = Vec::with_capacity(p);
    for &idx in &order {
        bal_eigenvalues_ordered.push(bal_eigenvalues[idx]);
    }
    let max_bal = bal_eigenvalues_ordered
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let rank_tol = if max_bal > 0.0 {
        max_bal * 1e-12
    } else {
        1e-12
    };
    let penalized_rank = bal_eigenvalues_ordered
        .iter()
        .take_while(|&&val| val > rank_tol)
        .count();

    let mut rs_transformed_base: Vec<Mat<f64>> = Vec::with_capacity(m);
    for rs in &rs_faer {
        let mut product = Mat::<f64>::zeros(rs.nrows(), qs.ncols());
        matmul(
            product.as_mut(),
            Accum::Replace,
            rs.as_ref(),
            qs.as_ref(),
            1.0,
            Par::Seq,
        );
        rs_transformed_base.push(product);
    }

    Ok(ReparamInvariant {
        qs_base: mat_to_array(&qs),
        rs_transformed_base: rs_transformed_base.iter().map(mat_to_array).collect(),
        penalized_rank,
        has_nonzero,
    })
}

/// Implements a fast, numerically stable reparameterization of the coefficient
/// space that preserves the conditioning benefits of Wood (2011) Appendix B
/// without the iterative similarity-transform loop.
///
/// The new strategy builds a lambda-independent “balanced” penalty matrix by
/// scaling each penalty to unit Frobenius norm, performs a single eigenvalue
/// decomposition to separate penalized and null-space directions, and then
/// whitens the penalized block using the current smoothing parameters. This
/// yields the same well-conditioned basis as the recursive algorithm while
/// avoiding its repeated \(O(q^3)\) eigendecompositions.
///
/// Each entry in `rs_list` is a `p × rank_k` penalty square root for penalty
/// `k`. The vector `lambdas` provides the smoothing parameters, and `layout`
/// defines the model’s coefficient structure. The function returns the
/// transformed penalties, the orthogonal basis, and log-determinant
/// information required by PIRLS.
pub(crate) fn stable_reparameterization(
    rs_list: &[Array2<f64>],
    lambdas: &[f64],
    p: usize,
) -> Result<ReparamResult, EstimationError> {
    let invariant = precompute_reparam_invariant(rs_list, p)?;
    stable_reparameterization_with_invariant(rs_list, lambdas, p, &invariant)
}

/// Apply stable reparameterization using precomputed lambda-invariant structures.
pub(crate) fn stable_reparameterization_with_invariant(
    rs_list: &[Array2<f64>],
    lambdas: &[f64],
    p: usize,
    invariant: &ReparamInvariant,
) -> Result<ReparamResult, EstimationError> {
    use std::cmp::Ordering;

    let m = rs_list.len();

    if lambdas.len() != m {
        return Err(EstimationError::ParameterConstraintViolation(format!(
            "Lambda count mismatch: expected {} lambdas for {} penalties, got {}",
            m,
            m,
            lambdas.len()
        )));
    }

    if invariant.rs_transformed_base.len() != m {
        return Err(EstimationError::LayoutError(format!(
            "Reparameterization invariant mismatch: expected {} penalties, got {}",
            m,
            invariant.rs_transformed_base.len()
        )));
    }

    if m == 0 {
        return Ok(ReparamResult {
            s_transformed: Array2::zeros((p, p)),
            log_det: 0.0,
            det1: Array1::zeros(0),
            qs: Array2::eye(p),
            rs_transformed: vec![],
            rs_transposed: vec![],
            e_transformed: Array2::zeros((0, p)),
            u_truncated: Array2::zeros((p, p)), // All modes truncated when no penalties
        });
    }

    if !invariant.has_nonzero {
        return Ok(ReparamResult {
            s_transformed: Array2::zeros((p, p)),
            log_det: 0.0,
            det1: Array1::zeros(m),
            qs: invariant.qs_base.clone(),
            rs_transformed: rs_list.to_vec(),
            rs_transposed: rs_list.iter().map(transpose_owned).collect(),
            e_transformed: Array2::zeros((0, p)),
            u_truncated: Array2::zeros((p, p)), // All modes truncated when zero penalty
        });
    }

    let mut qs = array_to_faer(&invariant.qs_base);
    let mut rs_transformed: Vec<Mat<f64>> = invariant
        .rs_transformed_base
        .iter()
        .map(array_to_faer)
        .collect();

    let penalized_rank = invariant.penalized_rank;

    let mut s_lambda = Mat::<f64>::zeros(p, p);
    for (lambda, rs_k) in lambdas.iter().zip(rs_transformed.iter()) {
        let s_k = penalty_from_root_faer(rs_k);
        for i in 0..p {
            for j in 0..p {
                s_lambda[(i, j)] += *lambda * s_k[(i, j)];
            }
        }
    }

    if penalized_rank > 0 {
        let mut range_block = Mat::<f64>::zeros(penalized_rank, penalized_rank);
        for i in 0..penalized_rank {
            for j in 0..penalized_rank {
                range_block[(i, j)] = s_lambda[(i, j)];
            }
        }
        let (range_eigenvalues, range_eigenvectors) =
            robust_eigh_faer(&range_block, Side::Lower, "range penalty block")?;

        let mut range_order: Vec<usize> = (0..penalized_rank).collect();
        range_order.sort_by(|&i, &j| {
            range_eigenvalues[j]
                .partial_cmp(&range_eigenvalues[i])
                .unwrap_or(Ordering::Equal)
                .then(i.cmp(&j))
        });

        let mut range_rotation = Mat::<f64>::zeros(penalized_rank, penalized_rank);
        for (col_idx, &idx) in range_order.iter().enumerate() {
            for row in 0..penalized_rank {
                range_rotation[(row, col_idx)] = range_eigenvectors[(row, idx)];
            }
        }

        let mut qs_subset = Mat::<f64>::zeros(p, penalized_rank);
        for row in 0..p {
            for col in 0..penalized_rank {
                qs_subset[(row, col)] = qs[(row, col)];
            }
        }
        let mut qs_range = Mat::<f64>::zeros(p, penalized_rank);
        matmul(
            qs_range.as_mut(),
            Accum::Replace,
            qs_subset.as_ref(),
            range_rotation.as_ref(),
            1.0,
            Par::Seq,
        );
        for row in 0..p {
            for col in 0..penalized_rank {
                qs[(row, col)] = qs_range[(row, col)];
            }
        }

        for rs in rs_transformed.iter_mut() {
            if rs.ncols() >= penalized_rank {
                let rows = rs.nrows();
                let mut rs_subset = Mat::<f64>::zeros(rows, penalized_rank);
                for i in 0..rows {
                    for j in 0..penalized_rank {
                        rs_subset[(i, j)] = rs[(i, j)];
                    }
                }
                let mut updated = Mat::<f64>::zeros(rows, penalized_rank);
                matmul(
                    updated.as_mut(),
                    Accum::Replace,
                    rs_subset.as_ref(),
                    range_rotation.as_ref(),
                    1.0,
                    Par::Seq,
                );
                for i in 0..rows {
                    for j in 0..penalized_rank {
                        rs[(i, j)] = updated[(i, j)];
                    }
                }
            }
        }
    }

    let mut s_transformed = Mat::<f64>::zeros(p, p);
    let mut s_k_transformed_cache: Vec<Mat<f64>> = Vec::with_capacity(m);
    for (lambda, rs_k) in lambdas.iter().zip(rs_transformed.iter()) {
        let s_k = penalty_from_root_faer(rs_k);
        for i in 0..p {
            for j in 0..p {
                s_transformed[(i, j)] += *lambda * s_k[(i, j)];
            }
        }
        s_k_transformed_cache.push(s_k);
    }

    let (s_eigenvalues_raw, s_eigenvectors) =
        robust_eigh_faer(&s_transformed, Side::Lower, "combined penalty matrix")?;

    // Use FIXED STRUCTURAL RANK to ensure C² smoothness of the LAML objective.
    //
    // The LAML objective includes log|S_λ|_+ (log-pseudo-determinant of the penalty).
    // The structural rank is determined at precompute time from the balanced penalties
    // and represents the number of truly penalized directions in the model geometry.
    //
    // CRITICAL: Using adaptive rank (based on current eigenvalues) creates DISCONTINUITIES
    // in the objective function. When an eigenvalue crosses the threshold, the rank jumps,
    // causing a step change in log|S|_+. This violates the C² assumption required by BFGS.
    //
    // To handle noise eigenvalues without discontinuities:
    // 1. Use FIXED structural rank (smooth, continuous objective)
    // 2. Clamp eigenvalues to a relative floor when computing log_det
    //    This bounds the contribution of noise without changing the rank.

    // Sort eigenvalues descending with their indices
    let mut sorted_eigs: Vec<(usize, f64)> = s_eigenvalues_raw
        .iter()
        .enumerate()
        .map(|(i, &ev)| (i, ev))
        .collect();
    sorted_eigs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Use FIXED structural rank from invariant (ensures smooth objective)
    let structural_rank = penalized_rank.min(sorted_eigs.len());
    let selected_eigs: Vec<(usize, f64)> =
        sorted_eigs.iter().take(structural_rank).cloned().collect();

    // Relative floor for log_det: clamp small eigenvalues to avoid ln(noise)
    // This is applied when computing log_det, NOT when selecting rank
    let max_eig = sorted_eigs.first().map(|(_, v)| *v).unwrap_or(1.0);
    let eigenvalue_floor = max_eig * 1e-12;

    // Extract truncated eigenvector indices (those NOT in selected_eigs)
    // These span the null space and are needed for gradient correction
    let truncated_indices: Vec<usize> = sorted_eigs
        .iter()
        .skip(structural_rank)
        .map(|&(idx, _)| idx)
        .collect();
    let truncated_count = truncated_indices.len();

    // Build u_truncated matrix (p × truncated_count)
    let mut u_truncated_mat = Mat::<f64>::zeros(p, truncated_count);
    for (col_out, &col_in) in truncated_indices.iter().enumerate() {
        for row in 0..p {
            u_truncated_mat[(row, col_out)] = s_eigenvectors[(row, col_in)];
        }
    }

    // Use relative floor for eigenvalue clamping (prevents ln(noise) without changing rank)
    // eigenvalue_floor = max_eig * 1e-12 ensures bounded contribution from noise modes

    let mut e_transformed_mat = Mat::<f64>::zeros(structural_rank, p);
    for (row_idx, &(eig_idx, eigenval)) in selected_eigs.iter().enumerate() {
        let safe_eigenval = eigenval.max(eigenvalue_floor);
        let sqrt_eigenval = safe_eigenval.sqrt();
        for row in 0..p {
            e_transformed_mat[(row_idx, row)] = s_eigenvectors[(row, eig_idx)] * sqrt_eigenval;
        }
    }

    // Clamp eigenvalues to floor when computing log_det (bounded noise contribution)
    let log_det: f64 = selected_eigs
        .iter()
        .map(|&(_, ev)| ev.max(eigenvalue_floor).ln())
        .sum();

    let mut det1_vec = vec![0.0; lambdas.len()];

    // Build S⁺ using the selected eigenvalues (structural rank)
    let mut s_plus = Mat::<f64>::zeros(p, p);
    for &(eig_idx, eigenval) in selected_eigs.iter() {
        if eigenval > eigenvalue_floor {
            let inv = 1.0 / eigenval;
            for i in 0..p {
                let vi = s_eigenvectors[(i, eig_idx)];
                for j in 0..p {
                    s_plus[(i, j)] += inv * vi * s_eigenvectors[(j, eig_idx)];
                }
            }
        }
    }

    for (k, lambda) in lambdas.iter().enumerate() {
        let mut product = Mat::<f64>::zeros(p, p);
        matmul(
            product.as_mut(),
            Accum::Replace,
            s_plus.as_ref(),
            s_k_transformed_cache[k].as_ref(),
            1.0,
            Par::Seq,
        );
        let mut trace = 0.0_f64;
        for i in 0..p {
            trace += product[(i, i)];
        }
        det1_vec[k] = *lambda * trace;
    }

    // Rebuild s_transformed from e_transformed to ensure rank consistency.
    //
    // The sum of λ*S_k may contain numerical noise modes (eigenvalues ~1e-15) that
    // become significant when λ is large (e.g., 10^12). These modes would appear in H
    // but are truncated from log|S|_+, creating a "phantom penalty" in the objective.
    //
    // By reconstructing s_transformed = E^T * E, we force the penalty matrix used
    // in H to have the EXACT same rank structure as the one used for log|S|_+.
    // Any mode truncated from the prior is now strictly zero in the Hessian
    // calculation, ensuring mathematical consistency of the gradients.
    let mut s_truncated = Mat::<f64>::zeros(p, p);
    matmul(
        s_truncated.as_mut(),
        Accum::Replace,
        e_transformed_mat.transpose(),
        e_transformed_mat.as_ref(),
        1.0,
        Par::Seq,
    );

    Ok(ReparamResult {
        s_transformed: mat_to_array(&s_truncated),
        log_det,
        det1: Array1::from(det1_vec),
        qs: mat_to_array(&qs),
        rs_transformed: rs_transformed.iter().map(mat_to_array).collect(),
        rs_transposed: rs_transformed
            .iter()
            .map(|mat| Array2::from_shape_fn((mat.ncols(), mat.nrows()), |(i, j)| mat[(j, i)]))
            .collect(),
        e_transformed: mat_to_array(&e_transformed_mat),
        u_truncated: mat_to_array(&u_truncated_mat),
    })
}

/// Minimal engine layout descriptor that avoids domain-specific layout coupling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineDims {
    pub p: usize,
    pub k: usize,
}

impl EngineDims {
    pub fn new(p: usize, k: usize) -> Self {
        Self { p, k }
    }
}

/// Engine-facing stable reparameterization API using only `(p, k)`.
pub fn stable_reparameterization_engine(
    rs_list: &[Array2<f64>],
    lambdas: &[f64],
    dims: EngineDims,
) -> Result<ReparamResult, EstimationError> {
    stable_reparameterization(rs_list, lambdas, dims.p)
}

/// Engine-facing stable reparameterization API with precomputed invariant using only `(p, k)`.
pub fn stable_reparameterization_with_invariant_engine(
    rs_list: &[Array2<f64>],
    lambdas: &[f64],
    dims: EngineDims,
    invariant: &ReparamInvariant,
) -> Result<ReparamResult, EstimationError> {
    stable_reparameterization_with_invariant(rs_list, lambdas, dims.p, invariant)
}

/// Result of the stable penalized least squares solve
#[derive(Clone)]
pub struct StablePLSResult {
    /// Solution vector beta
    pub beta: Array1<f64>,
    /// Final penalized Hessian matrix
    pub penalized_hessian: Array2<f64>,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Scale parameter estimate
    pub scale: f64,
}

/// Calculate the condition number of a matrix using singular value decomposition (SVD).
///
/// The condition number is the ratio of the largest to smallest singular value.
/// A high condition number indicates the matrix is close to singular and
/// solving linear systems with it may be numerically unstable.
///
/// # Arguments
/// * `matrix` - The matrix to analyze
///
/// # Returns
/// * `Ok(condition_number)` - The condition number (max_sv / min_sv)
/// * `Ok(f64::INFINITY)` - If the matrix is effectively singular (min_sv < 1e-12)
/// * `Err` - If SVD computation fails
pub fn calculate_condition_number(matrix: &Array2<f64>) -> Result<f64, FaerLinalgError> {
    // Use SVD for all matrices - the cost depends on number of coefficients (p), not samples (n)
    // For typical GAMs, p is much smaller than n, making SVD computationally feasible and reliable
    let (_, s, _) = matrix.svd(false, false)?;

    // Get max and min singular values
    let max_sv = s.iter().fold(0.0_f64, |max, &val| max.max(val));
    let min_sv = s.iter().fold(f64::INFINITY, |min, &val| min.min(val));

    // Check for effective singularity
    if min_sv < 1e-12 {
        return Ok(f64::INFINITY);
    }

    Ok(max_sv / min_sv)
}
