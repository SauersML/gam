use crate::basis::analyze_penalty_block;
use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerEigh, FaerLinalgError, FaerSvd};
use crate::smooth::{BlockwisePenalty, PenaltyStructureHint};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par, Side};
use ndarray::{Array1, Array2, ArrayViewMut2, Axis, s};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::collections::{BTreeMap, HashSet};
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
                let positive_offsets: HashSet<usize> = offsets
                    .iter()
                    .filter_map(|&off| (off >= 0).then_some(off as usize))
                    .collect();
                for (band, &offset) in bands.iter().zip(offsets.iter()) {
                    let off = offset.unsigned_abs() as usize;
                    if offset < 0 && positive_offsets.contains(&off) {
                        continue;
                    }
                    for (idx, &value) in band.iter().enumerate() {
                        let (i, j) = if offset >= 0 {
                            (idx, idx + off)
                        } else {
                            (idx + off, idx)
                        };
                        if i >= dim || j >= dim {
                            continue;
                        }
                        dense[[i, j]] = value;
                        dense[[j, i]] = value;
                    }
                }
                dense
            }
            PenaltyRepresentation::Kronecker { left, right } => {
                let (lrows, l_cols) = left.dim();
                let (rrows, r_cols) = right.dim();
                let mut result = Array2::zeros((lrows * rrows, l_cols * r_cols));
                for i in 0..lrows {
                    for j in 0..l_cols {
                        let scale = left[(i, j)];
                        if scale == 0.0 {
                            continue;
                        }
                        let mut block = result.slice_mut(s![
                            i * rrows..(i + 1) * rrows,
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
                let positive_offsets: HashSet<usize> = offsets
                    .iter()
                    .filter_map(|&off| (off >= 0).then_some(off as usize))
                    .collect();
                for (band, &offset) in bands.iter().zip(offsets.iter()) {
                    let off = offset.unsigned_abs() as usize;
                    if offset < 0 && positive_offsets.contains(&off) {
                        continue;
                    }
                    for (idx, &value) in band.iter().enumerate() {
                        let (i, j) = if offset >= 0 {
                            (idx, idx + off)
                        } else {
                            (idx + off, idx)
                        };
                        let Some(entry_ij) = dest.get_mut((i, j)) else {
                            continue;
                        };
                        *entry_ij += weight * value;
                        if i != j
                            && let Some(entry_ji) = dest.get_mut((j, i))
                        {
                            *entry_ji += weight * value;
                        }
                    }
                }
            }
            PenaltyRepresentation::Kronecker { left, right } => {
                let (lrows, l_cols) = left.dim();
                let (rrows, r_cols) = right.dim();
                for i in 0..lrows {
                    for j in 0..l_cols {
                        let scale = left[(i, j)] * weight;
                        if scale == 0.0 {
                            continue;
                        }
                        let mut block = dest.slice_mut(s![
                            i * rrows..(i + 1) * rrows,
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

fn array_to_faer(array: &Array2<f64>) -> Mat<f64> {
    let (rows, cols) = array.dim();
    Mat::from_fn(rows, cols, |i, j| array[[i, j]])
}

fn mat_to_array(mat: &Mat<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((mat.nrows(), mat.ncols()));
    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            out[[i, j]] = mat[(i, j)];
        }
    }
    out
}

fn mat_max_abs_element(matrix: MatRef<'_, f64>) -> f64 {
    let (rows, cols) = matrix.shape();
    let mut maxval = 0.0_f64;
    for i in 0..rows {
        for j in 0..cols {
            let val = matrix[(i, j)];
            if val.is_finite() {
                maxval = maxval.max(val.abs());
            }
        }
    }
    maxval
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

fn clamp_eigenvalues_for_stability(eigenvalues: &mut [f64], context: &str) {
    // Upstream basis builders are expected to construct PSD penalties.
    // This clamp is a downstream numerical cleanup step for machine-scale
    // spectral noise, not a semantic repair for materially indefinite inputs.
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
}

fn robust_eighwith_policy<M, V, E, Validate, Sanitize, EigCall, DiagScale, AddRidge, MapErr>(
    matrix: &M,
    context: &str,
    validate_input: Validate,
    sanitize: Sanitize,
    mut eig_call: EigCall,
    diag_scale: DiagScale,
    mut addridge_to_diag: AddRidge,
    map_error: MapErr,
) -> Result<(Vec<f64>, V), EstimationError>
where
    Validate: Fn(&M, &str) -> Result<(), EstimationError>,
    Sanitize: Fn(&M) -> M,
    EigCall: FnMut(&M) -> Result<(Vec<f64>, V), E>,
    DiagScale: Fn(&M) -> f64,
    AddRidge: FnMut(&mut M, f64),
    MapErr: Fn(E) -> EstimationError,
{
    validate_input(matrix, context)?;

    let mut candidate = sanitize(matrix);
    let mut ridge = 0.0_f64;

    for attempt in 0..4 {
        match eig_call(&candidate) {
            Ok((mut eigenvalues, eigenvectors)) => {
                clamp_eigenvalues_for_stability(&mut eigenvalues, context);
                return Ok((eigenvalues, eigenvectors));
            }
            Err(err) => {
                if attempt == 3 {
                    return Err(map_error(err));
                }

                let scale = diag_scale(&candidate);
                let base = if scale.is_finite() {
                    (scale * 1e-8).max(1e-10)
                } else {
                    1e-8
                };
                ridge = if ridge == 0.0 { base } else { ridge * 10.0 };
                addridge_to_diag(&mut candidate, ridge);

                log::warn!(
                    "{} eigendecomposition failed on attempt {}. Added ridge {:.3e} before retrying.",
                    context,
                    attempt + 1,
                    ridge
                );
            }
        }
    }

    unreachable!("robust_eighwith_policy should return or error within 4 attempts")
}

fn robust_eigh_faer(
    matrix: &Mat<f64>,
    side: Side,
    context: &str,
) -> Result<(Vec<f64>, Mat<f64>), EstimationError> {
    robust_eighwith_policy(
        matrix,
        context,
        |mat, ctx| {
            let (rows, cols) = mat.as_ref().shape();
            for i in 0..rows {
                for j in 0..cols {
                    let val = mat[(i, j)];
                    if !val.is_finite() {
                        let max_abs = mat_max_abs_element(mat.as_ref());
                        return Err(EstimationError::InvalidInput(format!(
                            "{} contains non-finite entries (max finite magnitude {:.3e})",
                            ctx, max_abs
                        )));
                    }
                }
            }
            Ok(())
        },
        sanitize_symmetric_faer,
        |candidate| {
            let eig = candidate.as_ref().self_adjoint_eigen(side)?;
            let diag = eig.S();
            let mut eigenvalues = Vec::with_capacity(diag.dim());
            for idx in 0..diag.dim() {
                eigenvalues.push(diag[idx]);
            }

            let vectors_ref = eig.U();
            let mut eigenvectors = Mat::<f64>::zeros(vectors_ref.nrows(), vectors_ref.ncols());
            for i in 0..vectors_ref.nrows() {
                for j in 0..vectors_ref.ncols() {
                    eigenvectors[(i, j)] = vectors_ref[(i, j)];
                }
            }
            Ok((eigenvalues, eigenvectors))
        },
        |candidate| {
            let mut scale = 0.0_f64;
            for idx in 0..candidate.nrows() {
                let val = candidate[(idx, idx)];
                if val.is_finite() {
                    scale = scale.max(val.abs());
                }
            }
            scale
        },
        |candidate, ridge| {
            for idx in 0..candidate.nrows() {
                candidate[(idx, idx)] += ridge;
            }
        },
        |err| EstimationError::EigendecompositionFailed(FaerLinalgError::SelfAdjointEigen(err)),
    )
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
    let matrix_faer = array_to_faer(matrix);
    let (eigenvalues, eigenvectors) = robust_eigh_faer(&matrix_faer, side, context)?;
    Ok((Array1::from_vec(eigenvalues), mat_to_array(&eigenvectors)))
}

#[derive(Debug, Clone, Copy)]
struct SubspaceLeakageMetrics {
    max_abs_sq: f64,
    max_rel_sq: f64,
    worst_penalty: usize,
    max_cross_gram_abs: f64,
}

fn assess_subspace_leakage(
    qs: &Mat<f64>,
    rs_transformed: &[Mat<f64>],
    structural_rank: usize,
    p: usize,
) -> SubspaceLeakageMetrics {
    let mut max_abs_sq = 0.0_f64;
    let mut max_rel_sq = 0.0_f64;
    let mut worst_penalty = 0usize;

    for (k, rs) in rs_transformed.iter().enumerate() {
        let rows = rs.nrows();
        let cols = rs.ncols().min(p);
        let null_start = structural_rank.min(cols);
        let mut abs_sq = 0.0_f64;
        let mut total_sq = 0.0_f64;
        for i in 0..rows {
            for j in 0..cols {
                let v = rs[(i, j)];
                let vv = v * v;
                total_sq += vv;
                if j >= null_start {
                    abs_sq += vv;
                }
            }
        }
        let rel_sq = if total_sq > 0.0 {
            abs_sq / total_sq
        } else {
            0.0
        };
        if rel_sq > max_rel_sq {
            max_rel_sq = rel_sq;
            worst_penalty = k;
        }
        max_abs_sq = max_abs_sq.max(abs_sq);
    }

    let mut max_cross_gram_abs = 0.0_f64;
    let null_count = p.saturating_sub(structural_rank);
    if structural_rank > 0 && null_count > 0 {
        for i in 0..structural_rank {
            for j in 0..null_count {
                let qn_col = structural_rank + j;
                let mut dot = 0.0_f64;
                for r in 0..p {
                    dot += qs[(r, i)] * qs[(r, qn_col)];
                }
                max_cross_gram_abs = max_cross_gram_abs.max(dot.abs());
            }
        }
    }

    SubspaceLeakageMetrics {
        max_abs_sq,
        max_rel_sq,
        worst_penalty,
        max_cross_gram_abs,
    }
}

fn compose_qs_from_split(q_pen: &Mat<f64>, q_null: &Mat<f64>, p: usize) -> Mat<f64> {
    let rank = q_pen.ncols();
    let null_count = q_null.ncols();
    let mut qs = Mat::<f64>::zeros(p, p);
    for i in 0..p {
        for j in 0..rank {
            qs[(i, j)] = q_pen[(i, j)];
        }
        for j in 0..null_count {
            qs[(i, rank + j)] = q_null[(i, j)];
        }
    }
    qs
}

/// Computes the Kronecker product A ⊗ B for penalty matrix construction.
/// This is used to create tensor product penalties that enforce smoothness
/// in multiple dimensions for interaction terms.
pub fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (arows, a_cols) = a.dim();
    let (brows, b_cols) = b.dim();
    if arows == 0 || a_cols == 0 || brows == 0 || b_cols == 0 {
        return Array2::zeros((arows * brows, a_cols * b_cols));
    }
    let mut result = Array2::zeros((arows * brows, a_cols * b_cols));

    result
        .axis_chunks_iter_mut(Axis(0), brows)
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row_block)| {
            let arow = a.row(i);
            let col_chunks = row_block.axis_chunks_iter_mut(Axis(1), b_cols);
            for (j, mut block) in col_chunks.into_iter().enumerate() {
                let aval = arow[j];
                if aval == 0.0 {
                    continue;
                }
                for (dest, &src) in block.iter_mut().zip(b.iter()) {
                    *dest = aval * src;
                }
            }
        });

    result
}

/// Result of the stable reparameterization algorithm from Wood (2011) Appendix B
#[derive(Clone)]
pub struct ReparamResult {
    /// Penalty matrix in TRANSFORMED coefficient coordinates.
    ///
    /// This must be compatible with `beta_transformed` and `X_transformed = X * Qs`.
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
    /// Lambda-dependent penalty square root in TRANSFORMED coordinates (rank x p matrix).
    /// This is used for applying the actual penalty in the least squares solve.
    pub e_transformed: Array2<f64>,
    /// Truncated eigenvectors (p × m where m = p - structural_rank).
    ///
    /// Coordinate frame note:
    /// - This matrix is stored in the TRANSFORMED coefficient frame (post-`Qs`),
    ///   i.e. it is compatible with `rs_transformed`, `beta_transformed`, and
    ///   transformed Hessians without additional coordinate mapping.
    ///
    /// These vectors span the structural null space used by positive-part
    /// log-determinant conventions.
    pub u_truncated: Array2<f64>,
    /// The rho-independent shrinkage ridge magnitude that was added to each
    /// eigenvalue of the penalized block. Zero means no shrinkage was applied.
    pub penalty_shrinkage_ridge: f64,
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
        let analysis = analyze_penalty_block(s).map_err(|err| {
            EstimationError::InvalidInput(format!("penalty canonicalization failed: {err}"))
        })?;
        assert!(
            analysis.rank > 0 || s.iter().all(|v| v.abs() <= analysis.tol),
            "inactive penalty block reached square-root construction"
        );

        // Reuse eigendecomposition from analyze_penalty_block — no double eigendecomp.
        let tolerance = analysis.tol;
        let rank_k = analysis.rank;

        if rank_k == 0 {
            // Zero penalty matrix - return 0 x p matrix (STANDARDIZED: rank x p)
            rs_list.push(Array2::zeros((0, p)));
            continue;
        }

        // STANDARDIZED: Create rank x p square root matrix where S = rs^T * rs
        // Each row is sqrt(eigenvalue) * eigenvector^T
        let mut rs = Array2::zeros((rank_k, p));
        let mut row_idx = 0;

        for (i, &eigenval) in analysis.eigenvalues.iter().enumerate() {
            if eigenval > tolerance {
                let sqrt_eigenval = eigenval.sqrt();
                let eigenvec = analysis.eigenvectors.column(i);
                // Each row of rs is sqrt(eigenvalue) * eigenvector^T
                rs.row_mut(row_idx).assign(&(&eigenvec * sqrt_eigenval));
                row_idx += 1;
            }
        }

        rs_list.push(rs);
    }

    Ok(rs_list)
}

// ---------------------------------------------------------------------------
// Block-scale spectral decomposition
// ---------------------------------------------------------------------------
//
// These functions operate at the natural block scale of each penalty
// (p_local × p_local) instead of inflating to p_total × p_total.
// Structure hints (Ridge, Kronecker) enable closed-form or factored
// eigendecomposition when available.

/// Compute a block-local square root and embed into global coordinates.
/// Returns a `rank_k × p_total` matrix where nonzero entries live in `col_range`.
fn block_local_root(
    col_range: &Range<usize>,
    local: &Array2<f64>,
    p_total: usize,
) -> Result<Array2<f64>, EstimationError> {
    let p_local = col_range.len();
    debug_assert_eq!(local.nrows(), p_local);
    debug_assert_eq!(local.ncols(), p_local);

    let analysis = analyze_penalty_block(local).map_err(|err| {
        EstimationError::InvalidInput(format!("block-local penalty analysis failed: {err}"))
    })?;

    if analysis.rank == 0 {
        return Ok(Array2::zeros((0, p_total)));
    }

    let rank_k = analysis.rank;
    let tolerance = analysis.tol;
    let mut rs = Array2::zeros((rank_k, p_total));
    let mut row_idx = 0;
    for (i, &eigenval) in analysis.eigenvalues.iter().enumerate() {
        if eigenval > tolerance {
            let sqrt_eigenval = eigenval.sqrt();
            let eigenvec = analysis.eigenvectors.column(i);
            for (j, &v) in eigenvec.iter().enumerate() {
                rs[[row_idx, col_range.start + j]] = sqrt_eigenval * v;
            }
            row_idx += 1;
        }
    }

    Ok(rs)
}

/// Ridge (scaled identity) root: trivial closed-form, no eigendecomposition.
/// Returns a `block_size × p_total` matrix.
fn ridge_root(
    col_range: &Range<usize>,
    scale: f64,
    p_total: usize,
) -> Array2<f64> {
    let block_size = col_range.len();
    if scale <= 0.0 || block_size == 0 {
        return Array2::zeros((0, p_total));
    }
    let sqrt_scale = scale.sqrt();
    let mut rs = Array2::zeros((block_size, p_total));
    for i in 0..block_size {
        rs[[i, col_range.start + i]] = sqrt_scale;
    }
    rs
}

/// Kronecker-factored root: eigendecompose each factor separately.
///
/// For factors `[F_0, F_1, ..., F_k]` where the product is `F_0 ⊗ F_1 ⊗ ... ⊗ F_k`,
/// the eigendecomposition of the Kronecker product is the Kronecker product of
/// the factor eigendecompositions. The root is `R_0 ⊗ R_1 ⊗ ... ⊗ R_k` where
/// each `R_j` is the square root of factor `F_j`.
///
/// Cost: O(Σ q_j³) instead of O((Π q_j)³).
fn kronecker_root(
    col_range: &Range<usize>,
    factors: &[Array2<f64>],
    p_total: usize,
) -> Result<Array2<f64>, EstimationError> {
    if factors.is_empty() {
        return Ok(Array2::zeros((0, p_total)));
    }

    // Eigendecompose each factor and build its square root.
    struct FactorRoot {
        root: Array2<f64>,  // rank_j × q_j
        rank: usize,
        dim: usize,
    }

    let mut factor_roots = Vec::with_capacity(factors.len());
    for (j, factor) in factors.iter().enumerate() {
        let q_j = factor.nrows();
        if q_j != factor.ncols() {
            return Err(EstimationError::InvalidInput(format!(
                "Kronecker factor {j} must be square, got {}x{}", factor.nrows(), factor.ncols()
            )));
        }

        // Check if this factor is an identity matrix (common: I ⊗ S_j ⊗ I).
        let is_identity = {
            let mut is_id = true;
            'outer: for r in 0..q_j {
                for c in 0..q_j {
                    let expected = if r == c { 1.0 } else { 0.0 };
                    if (factor[[r, c]] - expected).abs() > 1e-12 {
                        is_id = false;
                        break 'outer;
                    }
                }
            }
            is_id
        };

        if is_identity {
            // Identity factor: root is identity, rank = dim.
            factor_roots.push(FactorRoot {
                root: Array2::eye(q_j),
                rank: q_j,
                dim: q_j,
            });
            continue;
        }

        // General PSD factor: eigendecompose at O(q_j^3).
        let analysis = analyze_penalty_block(factor).map_err(|err| {
            EstimationError::InvalidInput(format!(
                "Kronecker factor {j} eigendecomposition failed: {err}"
            ))
        })?;

        if analysis.rank == 0 {
            // Zero factor → entire Kronecker product is zero.
            return Ok(Array2::zeros((0, p_total)));
        }

        let mut root_j = Array2::zeros((analysis.rank, q_j));
        let mut row_idx = 0;
        for (i, &eigenval) in analysis.eigenvalues.iter().enumerate() {
            if eigenval > analysis.tol {
                let sqrt_ev = eigenval.sqrt();
                let evec = analysis.eigenvectors.column(i);
                for (c, &v) in evec.iter().enumerate() {
                    root_j[[row_idx, c]] = sqrt_ev * v;
                }
                row_idx += 1;
            }
        }

        factor_roots.push(FactorRoot {
            root: root_j,
            rank: analysis.rank,
            dim: q_j,
        });
    }

    // Build the Kronecker product of factor roots iteratively.
    // Start with the first factor's root, then Kronecker-multiply with each subsequent.
    let mut kron_root = factor_roots[0].root.clone();
    for fr in &factor_roots[1..] {
        let (r1, c1) = kron_root.dim();
        let (r2, c2) = (fr.rank, fr.dim);
        let mut new_root = Array2::zeros((r1 * r2, c1 * c2));
        for i1 in 0..r1 {
            for i2 in 0..r2 {
                for j1 in 0..c1 {
                    for j2 in 0..c2 {
                        new_root[[i1 * r2 + i2, j1 * c2 + j2]] =
                            kron_root[[i1, j1]] * fr.root[[i2, j2]];
                    }
                }
            }
        }
        kron_root = new_root;
    }

    // Embed into global coordinates.
    let rank_total = kron_root.nrows();
    let block_dim = kron_root.ncols();
    debug_assert_eq!(block_dim, col_range.len());
    let mut rs = Array2::zeros((rank_total, p_total));
    rs.slice_mut(s![.., col_range.start..col_range.end])
        .assign(&kron_root);

    Ok(rs)
}

/// Compute penalty square roots from blockwise penalties at block scale.
///
/// This is the block-aware replacement for `compute_penalty_square_roots`.
/// Each penalty is eigendecomposed at its natural block size (p_local × p_local)
/// rather than the global size (p_total × p_total), with fast paths for
/// Ridge (closed-form) and Kronecker (per-factor) structures.
///
/// Returns `Vec<Array2<f64>>` of `rank_k × p_total` matrices — same format
/// as the existing `compute_penalty_square_roots`.
pub fn compute_penalty_square_roots_blockwise(
    penalties: &[BlockwisePenalty],
    p_total: usize,
) -> Result<Vec<Array2<f64>>, EstimationError> {
    let mut rs_list = Vec::with_capacity(penalties.len());

    for bp in penalties {
        let rs = match &bp.structure_hint {
            Some(PenaltyStructureHint::Ridge(scale)) => {
                ridge_root(&bp.col_range, *scale, p_total)
            }
            Some(PenaltyStructureHint::Kronecker(factors)) => {
                kronecker_root(&bp.col_range, factors, p_total)?
            }
            None => {
                block_local_root(&bp.col_range, &bp.local, p_total)?
            }
        };
        rs_list.push(rs);
    }

    Ok(rs_list)
}

/// Create a balanced penalty root from blockwise penalties at block scale.
///
/// When all penalties have non-overlapping col_ranges, the balanced sum is
/// block-diagonal and eigendecomposition is done per-block.
/// Falls back to the global path when penalties overlap.
pub fn create_balanced_penalty_root_blockwise(
    penalties: &[BlockwisePenalty],
    p_total: usize,
) -> Result<Array2<f64>, EstimationError> {
    if penalties.is_empty() {
        return Ok(Array2::zeros((0, p_total)));
    }

    // Group penalties by col_range to detect block-diagonal structure.
    // Key: (start, end), Value: list of local penalty matrices.
    let mut block_groups: BTreeMap<(usize, usize), Vec<&Array2<f64>>> = BTreeMap::new();
    for bp in penalties {
        let key = (bp.col_range.start, bp.col_range.end);
        block_groups.entry(key).or_default().push(&bp.local);
    }

    // Check for overlapping ranges. Ranges are non-overlapping if, when sorted
    // by start, each range's start >= the previous range's end.
    let ranges: Vec<(usize, usize)> = block_groups.keys().copied().collect();
    let mut overlapping = false;
    for i in 1..ranges.len() {
        if ranges[i].0 < ranges[i - 1].1 {
            overlapping = true;
            break;
        }
    }

    if overlapping {
        // Fall back to global path: materialize p×p matrices.
        let s_list: Vec<Array2<f64>> = penalties.iter().map(|bp| bp.to_global(p_total)).collect();
        return create_balanced_penalty_root(&s_list, p_total);
    }

    // Non-overlapping: process each block independently.
    let mut total_rank = 0usize;
    struct BlockRoot {
        col_range: Range<usize>,
        root: Array2<f64>,  // rank_b × block_dim
    }
    let mut block_roots = Vec::with_capacity(block_groups.len());

    for (&(start, end), locals) in &block_groups {
        let block_dim = end - start;
        let mut s_balanced = Array2::zeros((block_dim, block_dim));

        for local in locals {
            let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if frob_norm > 1e-12 {
                s_balanced.scaled_add(1.0 / frob_norm, local);
            }
        }

        let (eigenvalues, eigenvectors) =
            robust_eigh(&s_balanced, Side::Lower, "balanced penalty block")?;

        let max_eig = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));
        let tolerance = if max_eig > 0.0 { max_eig * 1e-12 } else { 1e-12 };
        let block_rank = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();

        if block_rank == 0 {
            continue;
        }

        let mut root = Array2::zeros((block_rank, block_dim));
        let mut row_idx = 0;
        for (i, &eigenval) in eigenvalues.iter().enumerate() {
            if eigenval > tolerance {
                let sqrt_ev = eigenval.sqrt();
                let evec = eigenvectors.column(i);
                root.row_mut(row_idx).assign(&(&evec * sqrt_ev));
                row_idx += 1;
            }
        }

        total_rank += block_rank;
        block_roots.push(BlockRoot {
            col_range: start..end,
            root,
        });
    }

    if total_rank == 0 {
        return Ok(Array2::zeros((0, p_total)));
    }

    // Assemble global balanced root: total_rank × p_total
    let mut eb = Array2::zeros((total_rank, p_total));
    let mut row_offset = 0;
    for br in &block_roots {
        let rank_b = br.root.nrows();
        eb.slice_mut(s![row_offset..(row_offset + rank_b), br.col_range.start..br.col_range.end])
            .assign(&br.root);
        row_offset += rank_b;
    }

    Ok(eb)
}

/// Precompute reparameterization invariant from blockwise penalties at block scale.
///
/// When all penalties have non-overlapping col_ranges, the balanced sum is
/// block-diagonal and the Q_pen/Q_null subspace split is computed per-block.
/// Falls back to the global path when penalties overlap.
pub fn precompute_reparam_invariant_blockwise(
    penalties: &[BlockwisePenalty],
    rs_list: &[Array2<f64>],
    p_total: usize,
) -> Result<ReparamInvariant, EstimationError> {
    use std::cmp::Ordering;

    let m = penalties.len();
    if m != rs_list.len() {
        return Err(EstimationError::LayoutError(format!(
            "penalties/rs_list length mismatch: {} vs {}",
            m, rs_list.len()
        )));
    }

    if m == 0 {
        return Ok(ReparamInvariant {
            split: SubspaceSplit::identity(p_total),
            rs_transformed_base: Vec::new(),
            has_nonzero: false,
            max_balanced_eigenvalue: 0.0,
        });
    }

    // Group penalties by col_range.
    struct PenaltyRef {
        global_index: usize,
        local: Array2<f64>,
    }
    let mut block_groups: BTreeMap<(usize, usize), Vec<PenaltyRef>> = BTreeMap::new();
    for (i, bp) in penalties.iter().enumerate() {
        let key = (bp.col_range.start, bp.col_range.end);
        block_groups.entry(key).or_default().push(PenaltyRef {
            global_index: i,
            local: bp.local.clone(),
        });
    }

    // Check for overlapping ranges.
    let ranges: Vec<(usize, usize)> = block_groups.keys().copied().collect();
    let mut overlapping = false;
    for i in 1..ranges.len() {
        if ranges[i].0 < ranges[i - 1].1 {
            overlapping = true;
            break;
        }
    }

    if overlapping {
        // Fall back to global path.
        return precompute_reparam_invariant(rs_list, p_total);
    }

    // Non-overlapping: process each block independently.
    // The global Q_s matrix is block-diagonal: for each block, we compute a local
    // Q_pen / Q_null split and embed it into the global matrix.

    // Columns NOT covered by any penalty range get identity treatment (unpenalized).
    let mut covered = vec![false; p_total];
    for bp in penalties {
        for j in bp.col_range.clone() {
            covered[j] = true;
        }
    }

    // Process each block.
    struct BlockInvariant {
        col_range: Range<usize>,
        q_pen_local: Array2<f64>,   // block_dim × pen_rank
        q_null_local: Array2<f64>,  // block_dim × null_count
        max_bal_eigenvalue: f64,
        // For each penalty in this block: its global index and local transformed root
        penalty_transforms: Vec<(usize, Array2<f64>)>,
    }

    let mut block_invariants = Vec::with_capacity(block_groups.len());
    let mut has_nonzero = false;
    let mut global_max_bal = 0.0_f64;

    for (&(start, end), refs) in &block_groups {
        let block_dim = end - start;

        // Build local balanced sum.
        let mut s_balanced_local = Array2::zeros((block_dim, block_dim));
        let mut block_has_nonzero = false;
        for pref in refs {
            let frob_norm = pref.local.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if frob_norm > 1e-12 {
                s_balanced_local.scaled_add(1.0 / frob_norm, &pref.local);
                block_has_nonzero = true;
            }
        }

        if !block_has_nonzero {
            // All penalties in this block are zero → identity split (all null).
            let mut transforms = Vec::new();
            for pref in refs {
                let rs_k = &rs_list[pref.global_index];
                // Extract the block columns and transform by identity.
                let rs_local = rs_k.slice(s![.., start..end]).to_owned();
                transforms.push((pref.global_index, rs_local));
            }
            block_invariants.push(BlockInvariant {
                col_range: start..end,
                q_pen_local: Array2::zeros((block_dim, 0)),
                q_null_local: Array2::eye(block_dim),
                max_bal_eigenvalue: 0.0,
                penalty_transforms: transforms,
            });
            continue;
        }

        has_nonzero = true;

        // Eigendecompose the local balanced penalty.
        let (bal_eigenvalues, bal_eigenvectors) =
            robust_eigh(&s_balanced_local, Side::Lower, "balanced penalty block")?;

        // Sort eigenvalues descending.
        let mut order: Vec<usize> = (0..block_dim).collect();
        order.sort_by(|&i, &j| {
            bal_eigenvalues[j]
                .partial_cmp(&bal_eigenvalues[i])
                .unwrap_or(Ordering::Equal)
                .then(i.cmp(&j))
        });

        let max_bal = order
            .iter()
            .map(|&idx| bal_eigenvalues[idx].abs())
            .fold(0.0_f64, f64::max);
        let rank_tol = if max_bal > 0.0 { max_bal * 1e-12 } else { 1e-12 };
        let penalized_rank = order
            .iter()
            .take_while(|&&idx| bal_eigenvalues[idx] > rank_tol)
            .count();
        let null_count = block_dim - penalized_rank;

        // Build local Q_pen and Q_null.
        let mut q_pen_local = Array2::zeros((block_dim, penalized_rank));
        let mut q_null_local = Array2::zeros((block_dim, null_count));
        for (col_idx, &idx) in order.iter().enumerate() {
            if col_idx < penalized_rank {
                for row in 0..block_dim {
                    q_pen_local[[row, col_idx]] = bal_eigenvectors[[row, idx]];
                }
            } else {
                let null_col = col_idx - penalized_rank;
                for row in 0..block_dim {
                    q_null_local[[row, null_col]] = bal_eigenvectors[[row, idx]];
                }
            }
        }

        // Build local Q_s = [Q_pen | Q_null] and transform each penalty root.
        let mut qs_local = Array2::zeros((block_dim, block_dim));
        for i in 0..block_dim {
            for j in 0..penalized_rank {
                qs_local[[i, j]] = q_pen_local[[i, j]];
            }
            for j in 0..null_count {
                qs_local[[i, penalized_rank + j]] = q_null_local[[i, j]];
            }
        }

        let mut transforms = Vec::new();
        for pref in refs {
            let rs_k = &rs_list[pref.global_index];
            // Extract block columns from the global root, multiply by local Q_s.
            let rs_local = rs_k.slice(s![.., start..end]);
            let rs_transformed_local = rs_local.dot(&qs_local);
            transforms.push((pref.global_index, rs_transformed_local));
        }

        global_max_bal = global_max_bal.max(max_bal);
        block_invariants.push(BlockInvariant {
            col_range: start..end,
            q_pen_local,
            q_null_local,
            max_bal_eigenvalue: max_bal,
            penalty_transforms: transforms,
        });
    }

    if !has_nonzero {
        return Ok(ReparamInvariant {
            split: SubspaceSplit::identity(p_total),
            rs_transformed_base: rs_list.to_vec(),
            has_nonzero: false,
            max_balanced_eigenvalue: 0.0,
        });
    }

    // Assemble global Q_pen and Q_null from block-local splits.
    // Uncovered columns (not in any penalty range) are fully unpenalized → go to Q_null.
    let uncovered_cols: Vec<usize> = (0..p_total).filter(|j| !covered[*j]).collect();
    let total_pen_rank: usize = block_invariants.iter().map(|bi| bi.q_pen_local.ncols()).sum();
    let total_null: usize =
        block_invariants.iter().map(|bi| bi.q_null_local.ncols()).sum() + uncovered_cols.len();

    let mut q_pen = Array2::zeros((p_total, total_pen_rank));
    let mut q_null = Array2::zeros((p_total, total_null));
    let mut pen_col = 0;
    let mut null_col = 0;

    for bi in &block_invariants {
        let start = bi.col_range.start;
        let pen_r = bi.q_pen_local.ncols();
        let null_r = bi.q_null_local.ncols();
        if pen_r > 0 {
            q_pen
                .slice_mut(s![start..(start + bi.q_pen_local.nrows()), pen_col..(pen_col + pen_r)])
                .assign(&bi.q_pen_local);
            pen_col += pen_r;
        }
        if null_r > 0 {
            q_null
                .slice_mut(s![
                    start..(start + bi.q_null_local.nrows()),
                    null_col..(null_col + null_r)
                ])
                .assign(&bi.q_null_local);
            null_col += null_r;
        }
    }
    // Uncovered columns → identity columns in Q_null.
    for &j in &uncovered_cols {
        q_null[[j, null_col]] = 1.0;
        null_col += 1;
    }

    let split = SubspaceSplit { q_pen, q_null };

    // Assemble global transformed roots: for each penalty, embed the local
    // transformed root into global coordinates. The root is rank_k × p_total,
    // with nonzero entries only in the block columns, pre-multiplied by local Q_s.
    let mut rs_transformed_base = vec![Array2::zeros((0, 0)); m];
    for bi in &block_invariants {
        for &(gi, ref rs_local_transformed) in &bi.penalty_transforms {
            let rank_k = rs_local_transformed.nrows();
            let cols = p_total;
            let mut rs_global = Array2::zeros((rank_k, cols));
            rs_global
                .slice_mut(s![.., bi.col_range.start..bi.col_range.end])
                .assign(rs_local_transformed);
            rs_transformed_base[gi] = rs_global;
        }
    }

    Ok(ReparamInvariant {
        split,
        rs_transformed_base,
        has_nonzero,
        max_balanced_eigenvalue: global_max_bal,
    })
}

// ---------------------------------------------------------------------------
// CanonicalPenalty — block-local processed penalty for the solver
// ---------------------------------------------------------------------------

/// A canonicalized penalty with block-local root, ready for the solver.
///
/// Instead of storing a full `p x p` penalty matrix, this stores only the
/// `rank x block_dim` root and the column range, enabling O(p_k^2) operations
/// instead of O(p^2).
#[derive(Clone, Debug)]
pub struct CanonicalPenalty {
    /// Square root matrix: S_k = root^T * root.
    /// Shape: `rank x block_dim` for block-local, `rank x p` for dense.
    pub root: Array2<f64>,
    /// Column range in the global coefficient vector [start..end).
    /// For dense penalties this is `0..p`.
    pub col_range: std::ops::Range<usize>,
    /// Full parameter dimension p.
    pub total_dim: usize,
    /// Structural nullity of the local penalty.
    pub nullity: usize,
    /// The symmetrized block-local penalty matrix (block_dim × block_dim).
    /// Cached at construction time to avoid recomputing root^T * root
    /// in hot paths (penalty assembly, trace products).
    pub local: Array2<f64>,
    /// Positive eigenvalues of the local penalty matrix (length = rank).
    /// Cached at construction time for REML logdet block-factored paths.
    pub positive_eigenvalues: Vec<f64>,
}

impl CanonicalPenalty {
    /// Numerical rank of this penalty.
    pub fn rank(&self) -> usize {
        self.root.nrows()
    }

    /// Block dimension (number of columns this penalty covers).
    pub fn block_dim(&self) -> usize {
        self.col_range.len()
    }

    /// Whether this penalty is block-local (col_range != 0..total_dim).
    pub fn is_block_local(&self) -> bool {
        self.col_range.start != 0 || self.col_range.end != self.total_dim
    }

    /// Return the cached local penalty matrix.
    /// Shape: `block_dim x block_dim`.
    pub fn local_penalty(&self) -> Array2<f64> {
        self.local.clone()
    }

    /// Accumulate lambda * S_k into a pre-allocated `p x p` target matrix.
    /// Only touches the block [col_range × col_range].
    pub fn accumulate_weighted(&self, target: &mut Array2<f64>, lambda: f64) {
        if lambda == 0.0 || self.rank() == 0 {
            return;
        }
        let r = &self.col_range;
        target
            .slice_mut(s![r.start..r.end, r.start..r.end])
            .scaled_add(lambda, &self.local);
    }

    /// Compute `scale * tr(M · S_k)` where M is a `p × p` dense matrix.
    /// Only reads `M[start..end, start..end]` — O(block_dim²) not O(p²).
    pub fn trace_product(&self, m: &Array2<f64>, scale: f64) -> f64 {
        if self.rank() == 0 || scale == 0.0 {
            return 0.0;
        }
        let r = &self.col_range;
        let m_block = m.slice(s![r.start..r.end, r.start..r.end]);
        let rm = self.root.dot(&m_block);
        scale * rm.iter().zip(self.root.iter()).map(|(&a, &b)| a * b).sum::<f64>()
    }

    /// Compute `scale * v^T S_k v` (quadratic form).
    /// Only reads `v[start..end]` — O(rank × block_dim) not O(rank × p).
    pub fn quadratic(&self, v: &Array1<f64>, scale: f64) -> f64 {
        if self.rank() == 0 || scale == 0.0 {
            return 0.0;
        }
        let v_block = v.slice(s![self.col_range.start..self.col_range.end]);
        let rv = self.root.dot(&v_block);
        scale * rv.dot(&rv)
    }

    /// Global root: `rank x p` matrix (embeds block root into full column space).
    pub fn global_root(&self) -> Array2<f64> {
        if !self.is_block_local() {
            return self.root.clone();
        }
        let mut g = Array2::zeros((self.rank(), self.total_dim));
        g.slice_mut(s![.., self.col_range.start..self.col_range.end])
            .assign(&self.root);
        g
    }

    /// Global penalty: `p x p` matrix (embeds into full space). Use sparingly.
    pub fn global_penalty(&self) -> Array2<f64> {
        if !self.is_block_local() {
            return self.local_penalty();
        }
        let local = self.local_penalty();
        let mut g = Array2::zeros((self.total_dim, self.total_dim));
        let r = &self.col_range;
        g.slice_mut(s![r.start..r.end, r.start..r.end])
            .assign(&local);
        g
    }

    /// Convert to a PenaltyCoordinate for the unified REML evaluator.
    pub fn to_penalty_coordinate(
        &self,
    ) -> crate::solver::estimate::reml::unified::PenaltyCoordinate {
        use crate::solver::estimate::reml::unified::PenaltyCoordinate;
        if self.is_block_local() {
            PenaltyCoordinate::BlockRoot {
                root: self.root.clone(),
                start: self.col_range.start,
                end: self.col_range.end,
                total_dim: self.total_dim,
            }
        } else {
            PenaltyCoordinate::from_dense_root(self.root.clone())
        }
    }
}

/// Canonicalize a single `PenaltySpec` into a `CanonicalPenalty` by computing
/// the block-local eigendecomposition and extracting the root.
///
/// This is O(block_dim^3) instead of O(p^3) for block-local penalties.
/// Returns `None` if the penalty has rank zero (should be dropped).
pub fn canonicalize_penalty_spec(
    spec: &crate::estimate::PenaltySpec,
    p: usize,
    idx: usize,
    context: &str,
) -> Result<Option<CanonicalPenalty>, EstimationError> {
    use crate::estimate::PenaltySpec;

    let (local_matrix, col_range) = match spec {
        PenaltySpec::Block { local, col_range } => {
            let bd = col_range.len();
            if local.nrows() != bd || local.ncols() != bd {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: block penalty {idx} local matrix must be {bd}x{bd}, got {}x{}",
                    local.nrows(),
                    local.ncols()
                )));
            }
            if col_range.end > p {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: block penalty {idx} col_range {}..{} exceeds p={p}",
                    col_range.start, col_range.end
                )));
            }
            (local.view(), col_range.clone())
        }
        PenaltySpec::Dense(m) => {
            if m.nrows() != p || m.ncols() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                    m.nrows(),
                    m.ncols()
                )));
            }
            (m.view(), 0..p)
        }
    };

    let local_owned = local_matrix.to_owned();
    let analysis = analyze_penalty_block(&local_owned).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "{context}: penalty canonicalization failed at index {idx}: {err}"
        ))
    })?;

    if analysis.rank == 0 {
        log::debug!(
            "Dropped inactive penalty block idx={idx} reason={}",
            if analysis.iszero {
                "ZeroMatrix"
            } else {
                "NumericalRankZero"
            }
        );
        return Ok(None);
    }

    // Reuse eigendecomposition from analyze_penalty_block — no double eigendecomp.
    let tolerance = analysis.tol;
    let rank_k = analysis.rank;
    let block_dim = local_owned.nrows();

    let mut root = Array2::zeros((rank_k, block_dim));
    let mut positive_eigenvalues = Vec::with_capacity(rank_k);
    let mut row_idx = 0;
    for (i, &eigenval) in analysis.eigenvalues.iter().enumerate() {
        if eigenval > tolerance {
            let sqrt_eigenval = eigenval.sqrt();
            let eigenvec = analysis.eigenvectors.column(i);
            root.row_mut(row_idx).assign(&(&eigenvec * sqrt_eigenval));
            positive_eigenvalues.push(eigenval);
            row_idx += 1;
        }
    }

    Ok(Some(CanonicalPenalty {
        root,
        col_range,
        total_dim: p,
        nullity: analysis.nullity,
        local: analysis.sym_penalty,
        positive_eigenvalues,
    }))
}

/// Canonicalize a batch of penalty specs, dropping zero-rank penalties.
/// Returns (active_penalties, active_nullspace_dims).
pub fn canonicalize_penalty_specs(
    specs: &[crate::estimate::PenaltySpec],
    nullspace_dims: &[usize],
    p: usize,
    context: &str,
) -> Result<(Vec<CanonicalPenalty>, Vec<usize>), EstimationError> {
    if specs.len() != nullspace_dims.len() {
        return Err(EstimationError::InvalidInput(format!(
            "{context}: nullspace_dims length mismatch: penalties={}, nullspace_dims={}",
            specs.len(),
            nullspace_dims.len()
        )));
    }

    let mut active = Vec::with_capacity(specs.len());
    let mut active_nullspace = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        if let Some(canonical) = canonicalize_penalty_spec(spec, p, idx, context)? {
            active_nullspace.push(canonical.nullity);
            active.push(canonical);
        }
    }
    Ok((active, active_nullspace))
}

/// Creates a balanced penalty root from canonical penalties.
///
/// When all penalties have non-overlapping col_ranges, the balanced sum is
/// block-diagonal and eigendecomposition is done per-block at O(Σ p_k³)
/// instead of the global O(p³). Falls back to the global path when penalties
/// overlap.
pub fn create_balanced_penalty_root_from_canonical(
    penalties: &[CanonicalPenalty],
    p: usize,
) -> Result<Array2<f64>, EstimationError> {
    if penalties.is_empty() {
        return Ok(Array2::zeros((0, p)));
    }

    // Group penalties by col_range.
    let mut block_groups: BTreeMap<(usize, usize), Vec<&CanonicalPenalty>> = BTreeMap::new();
    for cp in penalties {
        if cp.rank() == 0 {
            continue;
        }
        let key = (cp.col_range.start, cp.col_range.end);
        block_groups.entry(key).or_default().push(cp);
    }

    if block_groups.is_empty() {
        return Ok(Array2::zeros((0, p)));
    }

    // Check for overlapping ranges.
    let ranges: Vec<(usize, usize)> = block_groups.keys().copied().collect();
    let mut overlapping = false;
    for i in 1..ranges.len() {
        if ranges[i].0 < ranges[i - 1].1 {
            overlapping = true;
            break;
        }
    }

    if overlapping {
        // Fallback: accumulate into p × p and eigendecompose globally.
        let mut s_balanced = Array2::zeros((p, p));
        for cp in penalties {
            if cp.rank() == 0 {
                continue;
            }
            let local = cp.local_penalty();
            let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if frob_norm > 1e-12 {
                let r = &cp.col_range;
                s_balanced
                    .slice_mut(s![r.start..r.end, r.start..r.end])
                    .scaled_add(1.0 / frob_norm, &local);
            }
        }
        let (eigenvalues, eigenvectors) =
            robust_eigh(&s_balanced, Side::Lower, "balanced penalty matrix")?;
        let max_eig = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));
        let tolerance = if max_eig > 0.0 { max_eig * 1e-12 } else { 1e-12 };
        let penalty_rank = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();
        if penalty_rank == 0 {
            return Ok(Array2::zeros((0, p)));
        }
        let mut eb = Array2::zeros((p, penalty_rank));
        let mut col_idx = 0;
        for (i, &eigenval) in eigenvalues.iter().enumerate() {
            if eigenval > tolerance {
                let sqrt_ev = eigenval.sqrt();
                let evec = eigenvectors.column(i);
                eb.column_mut(col_idx).assign(&(&evec * sqrt_ev));
                col_idx += 1;
            }
        }
        return Ok(eb.t().to_owned());
    }

    // Non-overlapping: eigendecompose per block at O(Σ p_k³).
    struct BlockRoot {
        col_range: Range<usize>,
        root: Array2<f64>,  // rank_b × block_dim
    }
    let mut total_rank = 0usize;
    let mut block_roots = Vec::with_capacity(block_groups.len());

    for (&(start, end), cps) in &block_groups {
        let block_dim = end - start;
        let mut s_balanced_local = Array2::zeros((block_dim, block_dim));

        for cp in cps {
            let local = cp.local_penalty();
            let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if frob_norm > 1e-12 {
                s_balanced_local.scaled_add(1.0 / frob_norm, &local);
            }
        }

        let (eigenvalues, eigenvectors) =
            robust_eigh(&s_balanced_local, Side::Lower, "balanced penalty block")?;
        let max_eig = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));
        let tolerance = if max_eig > 0.0 { max_eig * 1e-12 } else { 1e-12 };
        let block_rank = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();

        if block_rank == 0 {
            continue;
        }

        let mut root = Array2::zeros((block_rank, block_dim));
        let mut row_idx = 0;
        for (i, &eigenval) in eigenvalues.iter().enumerate() {
            if eigenval > tolerance {
                let sqrt_ev = eigenval.sqrt();
                let evec = eigenvectors.column(i);
                root.row_mut(row_idx).assign(&(&evec * sqrt_ev));
                row_idx += 1;
            }
        }

        total_rank += block_rank;
        block_roots.push(BlockRoot {
            col_range: start..end,
            root,
        });
    }

    if total_rank == 0 {
        return Ok(Array2::zeros((0, p)));
    }

    // Assemble global balanced root: total_rank × p
    let mut eb = Array2::zeros((total_rank, p));
    let mut row_offset = 0;
    for br in &block_roots {
        let rank_b = br.root.nrows();
        eb.slice_mut(s![row_offset..(row_offset + rank_b), br.col_range.start..br.col_range.end])
            .assign(&br.root);
        row_offset += rank_b;
    }

    Ok(eb)
}

/// Lambda-independent reparameterization invariants derived from penalty structure.
#[derive(Clone)]
struct SubspaceSplit {
    q_pen: Array2<f64>,
    q_null: Array2<f64>,
}

impl SubspaceSplit {
    fn identity(p: usize) -> Self {
        Self {
            q_pen: Array2::zeros((p, 0)),
            q_null: Array2::eye(p),
        }
    }

    fn from_ordered_qs(
        qs: &Mat<f64>,
        penalized_rank: usize,
        p: usize,
    ) -> Result<Self, EstimationError> {
        if qs.nrows() != p || qs.ncols() != p {
            return Err(EstimationError::LayoutError(format!(
                "Invalid Q basis dimensions: expected {p}x{p}, got {}x{}",
                qs.nrows(),
                qs.ncols()
            )));
        }
        if penalized_rank > p {
            return Err(EstimationError::LayoutError(format!(
                "Invalid penalized rank {penalized_rank} for p={p}"
            )));
        }

        let null_count = p - penalized_rank;
        let mut q_pen = Array2::<f64>::zeros((p, penalized_rank));
        let mut q_null = Array2::<f64>::zeros((p, null_count));
        for i in 0..p {
            for j in 0..penalized_rank {
                q_pen[(i, j)] = qs[(i, j)];
            }
            for j in 0..null_count {
                q_null[(i, j)] = qs[(i, penalized_rank + j)];
            }
        }

        Ok(Self { q_pen, q_null })
    }

    fn rank(&self) -> usize {
        self.q_pen.ncols()
    }

    fn p(&self) -> usize {
        self.q_pen.nrows()
    }

    fn compose_qs(&self) -> Array2<f64> {
        let p = self.p();
        let rank = self.rank();
        let null_count = self.q_null.ncols();
        let mut qs = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..rank {
                qs[(i, j)] = self.q_pen[(i, j)];
            }
            for j in 0..null_count {
                qs[(i, rank + j)] = self.q_null[(i, j)];
            }
        }
        qs
    }
}

/// Lambda-independent reparameterization invariants derived from penalty structure.
#[derive(Clone)]
pub struct ReparamInvariant {
    split: SubspaceSplit,
    rs_transformed_base: Vec<Array2<f64>>,
    has_nonzero: bool,
    /// Largest eigenvalue of the balanced (unit-Frobenius) penalty matrix.
    /// Used as the scale reference for the shrinkage floor.
    max_balanced_eigenvalue: f64,
}

impl ReparamInvariant {
    /// Returns the largest eigenvalue of the balanced penalty matrix.
    /// This is lambda-independent and provides a natural scale for shrinkage.
    pub fn max_balanced_eigenvalue(&self) -> f64 {
        self.max_balanced_eigenvalue
    }
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
            split: SubspaceSplit::identity(p),
            rs_transformed_base: Vec::new(),
            has_nonzero: false,
            max_balanced_eigenvalue: 0.0,
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
            split: SubspaceSplit::identity(p),
            rs_transformed_base: rs_list.to_vec(),
            has_nonzero: false,
            max_balanced_eigenvalue: 0.0,
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
    let split = SubspaceSplit::from_ordered_qs(&qs, penalized_rank, p)?;

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
        split,
        rs_transformed_base: rs_transformed_base.iter().map(mat_to_array).collect(),
        has_nonzero,
        max_balanced_eigenvalue: max_bal,
    })
}

/// Precompute the lambda-invariant reparameterization structure from canonical penalties.
/// Same algorithm as `precompute_reparam_invariant`, but uses block-local roots
/// directly instead of requiring rank x p global roots.
pub fn precompute_reparam_invariant_from_canonical(
    penalties: &[CanonicalPenalty],
    p: usize,
) -> Result<ReparamInvariant, EstimationError> {
    use std::cmp::Ordering;

    let m = penalties.len();

    if m == 0 {
        return Ok(ReparamInvariant {
            split: SubspaceSplit::identity(p),
            rs_transformed_base: Vec::new(),
            has_nonzero: false,
            max_balanced_eigenvalue: 0.0,
        });
    }

    // Group penalties by col_range to detect block-diagonal structure.
    struct PenRef {
        penalty_index: usize,
    }
    let mut block_groups: BTreeMap<(usize, usize), Vec<PenRef>> = BTreeMap::new();
    let mut has_nonzero = false;
    for (i, cp) in penalties.iter().enumerate() {
        if cp.rank() == 0 {
            continue;
        }
        let local = cp.local_penalty();
        let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if frob_norm > 1e-12 {
            has_nonzero = true;
        }
        let key = (cp.col_range.start, cp.col_range.end);
        block_groups.entry(key).or_default().push(PenRef { penalty_index: i });
    }

    if !has_nonzero {
        let global_roots: Vec<Array2<f64>> = penalties.iter().map(|cp| cp.global_root()).collect();
        return Ok(ReparamInvariant {
            split: SubspaceSplit::identity(p),
            rs_transformed_base: global_roots,
            has_nonzero: false,
            max_balanced_eigenvalue: 0.0,
        });
    }

    // Check for overlapping ranges.
    let ranges: Vec<(usize, usize)> = block_groups.keys().copied().collect();
    let mut overlapping = false;
    for i in 1..ranges.len() {
        if ranges[i].0 < ranges[i - 1].1 {
            overlapping = true;
            break;
        }
    }

    if overlapping {
        // Fallback: global p×p eigendecomposition.
        let global_roots: Vec<Array2<f64>> =
            penalties.iter().map(|cp| cp.global_root()).collect();
        let rs_faer: Vec<Mat<f64>> = global_roots.iter().map(array_to_faer).collect();

        let mut s_balanced = Mat::<f64>::zeros(p, p);
        for cp in penalties {
            if cp.rank() == 0 {
                continue;
            }
            let local = cp.local_penalty();
            let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if frob_norm > 1e-12 {
                let scale = 1.0 / frob_norm;
                let r = &cp.col_range;
                for i in 0..local.nrows() {
                    for j in 0..local.ncols() {
                        s_balanced[(r.start + i, r.start + j)] += scale * local[[i, j]];
                    }
                }
            }
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

        let max_bal = order
            .iter()
            .map(|&idx| bal_eigenvalues[idx].abs())
            .fold(0.0_f64, f64::max);
        let rank_tol = if max_bal > 0.0 { max_bal * 1e-12 } else { 1e-12 };
        let penalized_rank = order
            .iter()
            .take_while(|&&idx| bal_eigenvalues[idx] > rank_tol)
            .count();
        let split = SubspaceSplit::from_ordered_qs(&qs, penalized_rank, p)?;

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

        return Ok(ReparamInvariant {
            split,
            rs_transformed_base: rs_transformed_base.iter().map(mat_to_array).collect(),
            has_nonzero,
            max_balanced_eigenvalue: max_bal,
        });
    }

    // -----------------------------------------------------------------------
    // Non-overlapping: block-diagonal eigendecomposition at O(Σ p_k³).
    // -----------------------------------------------------------------------
    // The balanced sum is block-diagonal ⟹ its eigenvectors are block-local.
    // Q_pen and Q_null are assembled by embedding block-local eigenvectors.

    // Track which columns are covered by any penalty.
    let mut covered = vec![false; p];
    for cp in penalties {
        for j in cp.col_range.clone() {
            covered[j] = true;
        }
    }
    let uncovered_cols: Vec<usize> = (0..p).filter(|j| !covered[*j]).collect();

    struct BlockResult {
        col_range: Range<usize>,
        q_pen_local: Array2<f64>,
        q_null_local: Array2<f64>,
        qs_local: Array2<f64>,
        max_bal: f64,
        penalty_indices: Vec<usize>,
    }

    let mut block_results = Vec::with_capacity(block_groups.len());
    let mut global_max_bal = 0.0_f64;

    for (&(start, end), refs) in &block_groups {
        let block_dim = end - start;

        // Build local balanced sum.
        let mut s_balanced_local = Array2::zeros((block_dim, block_dim));
        let mut block_has_nonzero = false;
        for pref in refs {
            let cp = &penalties[pref.penalty_index];
            let local = cp.local_penalty();
            let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if frob_norm > 1e-12 {
                s_balanced_local.scaled_add(1.0 / frob_norm, &local);
                block_has_nonzero = true;
            }
        }

        let penalty_indices: Vec<usize> = refs.iter().map(|r| r.penalty_index).collect();

        if !block_has_nonzero {
            block_results.push(BlockResult {
                col_range: start..end,
                q_pen_local: Array2::zeros((block_dim, 0)),
                q_null_local: Array2::eye(block_dim),
                qs_local: Array2::eye(block_dim),
                max_bal: 0.0,
                penalty_indices,
            });
            continue;
        }

        // Eigendecompose the local balanced penalty.
        let (bal_eigenvalues, bal_eigenvectors) =
            robust_eigh(&s_balanced_local, Side::Lower, "balanced penalty block")?;

        let mut order: Vec<usize> = (0..block_dim).collect();
        order.sort_by(|&i, &j| {
            bal_eigenvalues[j]
                .partial_cmp(&bal_eigenvalues[i])
                .unwrap_or(Ordering::Equal)
                .then(i.cmp(&j))
        });

        let max_bal = order
            .iter()
            .map(|&idx| bal_eigenvalues[idx].abs())
            .fold(0.0_f64, f64::max);
        let rank_tol = if max_bal > 0.0 { max_bal * 1e-12 } else { 1e-12 };
        let penalized_rank = order
            .iter()
            .take_while(|&&idx| bal_eigenvalues[idx] > rank_tol)
            .count();
        let null_count = block_dim - penalized_rank;

        let mut q_pen_local = Array2::zeros((block_dim, penalized_rank));
        let mut q_null_local = Array2::zeros((block_dim, null_count));
        for (col_idx, &idx) in order.iter().enumerate() {
            if col_idx < penalized_rank {
                for row in 0..block_dim {
                    q_pen_local[[row, col_idx]] = bal_eigenvectors[[row, idx]];
                }
            } else {
                let null_col = col_idx - penalized_rank;
                for row in 0..block_dim {
                    q_null_local[[row, null_col]] = bal_eigenvectors[[row, idx]];
                }
            }
        }

        // Local Q_s = [Q_pen | Q_null]
        let mut qs_local = Array2::zeros((block_dim, block_dim));
        for i in 0..block_dim {
            for j in 0..penalized_rank {
                qs_local[[i, j]] = q_pen_local[[i, j]];
            }
            for j in 0..null_count {
                qs_local[[i, penalized_rank + j]] = q_null_local[[i, j]];
            }
        }

        global_max_bal = global_max_bal.max(max_bal);
        block_results.push(BlockResult {
            col_range: start..end,
            q_pen_local,
            q_null_local,
            qs_local,
            max_bal,
            penalty_indices,
        });
    }

    // Assemble global Q_pen and Q_null.
    let total_pen_rank: usize = block_results.iter().map(|br| br.q_pen_local.ncols()).sum();
    let total_null: usize =
        block_results.iter().map(|br| br.q_null_local.ncols()).sum() + uncovered_cols.len();

    let mut q_pen = Array2::zeros((p, total_pen_rank));
    let mut q_null = Array2::zeros((p, total_null));
    let mut pen_col = 0usize;
    let mut null_col = 0usize;

    for br in &block_results {
        let start = br.col_range.start;
        let bd = br.q_pen_local.nrows();
        let pen_r = br.q_pen_local.ncols();
        let null_r = br.q_null_local.ncols();
        if pen_r > 0 {
            q_pen
                .slice_mut(s![start..(start + bd), pen_col..(pen_col + pen_r)])
                .assign(&br.q_pen_local);
            pen_col += pen_r;
        }
        if null_r > 0 {
            q_null
                .slice_mut(s![start..(start + bd), null_col..(null_col + null_r)])
                .assign(&br.q_null_local);
            null_col += null_r;
        }
    }
    for &j in &uncovered_cols {
        q_null[[j, null_col]] = 1.0;
        null_col += 1;
    }

    let split = SubspaceSplit { q_pen, q_null };

    // Assemble global transformed roots.
    // For each penalty, extract the block columns from its global root and
    // multiply by the local Q_s. Since Q_s is block-diagonal, this only
    // touches the relevant block columns.
    let mut rs_transformed_base = vec![Array2::zeros((0, p)); m];

    for br in &block_results {
        let start = br.col_range.start;
        let end = br.col_range.end;

        for &pi in &br.penalty_indices {
            let cp = &penalties[pi];
            // Extract local root: root is rank_k × block_dim.
            let local_root = &cp.root;
            let rank_k = local_root.nrows();

            // Transform: rs_local_transformed = local_root * qs_local
            let rs_transformed_local = local_root.dot(&br.qs_local);

            // Embed into global coordinates: rank_k × p.
            let mut rs_global = Array2::zeros((rank_k, p));
            rs_global
                .slice_mut(s![.., start..end])
                .assign(&rs_transformed_local);
            rs_transformed_base[pi] = rs_global;
        }
    }

    // Handle penalties with rank 0 that didn't enter any block group.
    for (i, cp) in penalties.iter().enumerate() {
        if rs_transformed_base[i].dim() == (0, p) {
            continue; // already sized correctly for rank-0
        }
        if rs_transformed_base[i].nrows() == 0 && rs_transformed_base[i].ncols() == 0 {
            // Rank-0 penalty that wasn't in any block group.
            rs_transformed_base[i] = Array2::zeros((cp.rank(), p));
        }
    }

    Ok(ReparamInvariant {
        split,
        rs_transformed_base,
        has_nonzero,
        max_balanced_eigenvalue: global_max_bal,
    })
}

/// Apply stable reparameterization using precomputed lambda-invariant structures.
///
/// `penalty_shrinkage_floor`: optional relative shrinkage floor for eigenvalues
/// of the penalized block. If `Some(epsilon)`, a rho-independent ridge of
/// magnitude `epsilon * max_balanced_eigenvalue` is added to each eigenvalue
/// of the combined penalty on the penalized block. This prevents barely-penalized
/// directions from causing pathological non-Gaussianity in the posterior (e.g.,
/// extreme skewness under logit link with high-dimensional spatial smooths).
/// A typical value is `1e-6`. Set to `None` or `Some(0.0)` to disable.
pub fn stable_reparameterizationwith_invariant(
    rs_list: &[Array2<f64>],
    lambdas: &[f64],
    p: usize,
    invariant: &ReparamInvariant,
    penalty_shrinkage_floor: Option<f64>,
) -> Result<ReparamResult, EstimationError> {
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
            // All modes truncated when no penalties; already in transformed frame.
            u_truncated: Array2::eye(p),
            penalty_shrinkage_ridge: 0.0,
        });
    }

    if !invariant.has_nonzero {
        let qs = invariant.split.compose_qs();
        let u_truncated = qs.t().dot(&invariant.split.q_null);
        return Ok(ReparamResult {
            s_transformed: Array2::zeros((p, p)),
            log_det: 0.0,
            det1: Array1::zeros(m),
            qs,
            rs_transformed: rs_list.to_vec(),
            rs_transposed: rs_list.iter().map(transpose_owned).collect(),
            e_transformed: Array2::zeros((0, p)),
            // Stored in transformed frame for downstream trace/correction math.
            u_truncated, // All modes truncated when zero penalty
            penalty_shrinkage_ridge: 0.0,
        });
    }

    let q_pen = array_to_faer(&invariant.split.q_pen);
    let q_null = array_to_faer(&invariant.split.q_null);
    let rs_transformed: Vec<Mat<f64>> = invariant
        .rs_transformed_base
        .iter()
        .map(array_to_faer)
        .collect();

    let penalized_rank = invariant.split.rank();

    let mut range_eigenvalues_sorted: Vec<f64> = Vec::new();
    let mut range_rotation = Mat::<f64>::zeros(penalized_rank, penalized_rank);
    if penalized_rank > 0 {
        let mut range_block = Mat::<f64>::zeros(penalized_rank, penalized_rank);
        for (lambda, rs_k) in lambdas.iter().zip(rs_transformed.iter()) {
            let s_k = penalty_from_root_faer(rs_k);
            for i in 0..penalized_rank {
                for j in 0..penalized_rank {
                    range_block[(i, j)] += *lambda * s_k[(i, j)];
                }
            }
        }
        let (range_eigenvalues, range_eigenvectors) =
            robust_eigh_faer(&range_block, Side::Lower, "range penalty block")?;

        let mut range_order: Vec<usize> = (0..penalized_rank).collect();
        range_order.sort_by(|&i, &j| {
            range_eigenvalues[j]
                .partial_cmp(&range_eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(&j))
        });
        range_eigenvalues_sorted = range_order
            .iter()
            .map(|&idx| range_eigenvalues[idx])
            .collect();

        // Build range_rotation = U (sorted eigenvectors) for E and S⁺
        // construction only.  DO NOT apply to q_pen or rs_transformed —
        // keeping Q_s lambda-independent prevents BFGS coordinate-system
        // drift when multiple penalties interact (the eigenvectors of
        // Σ λ_k S_k rotate with λ, breaking the quasi-Newton Hessian
        // approximation at eigenvalue crossings).
        for (col_idx, &idx) in range_order.iter().enumerate() {
            for row in 0..penalized_rank {
                range_rotation[(row, col_idx)] = range_eigenvectors[(row, idx)];
            }
        }
        // q_pen and rs_transformed stay in the lambda-independent
        // invariant basis.  E and S⁺ below are expressed in this same
        // basis using U from the eigendecomposition.
    }

    let mut s_k_transformed_cache: Vec<Mat<f64>> = Vec::with_capacity(m);
    for rs_k in rs_transformed.iter() {
        let s_k = penalty_from_root_faer(rs_k);
        s_k_transformed_cache.push(s_k);
    }

    // Subspace-invariant penalty spectral calculus:
    // - Penalized and null spaces are fixed by the lambda-invariant basis `qs_base`.
    // - Runtime lambda dependence only appears in the penalized block eigenvalues.
    // This avoids basis mixing inside the degenerate zero-eigenspace.
    let structural_rank = penalized_rank;
    let mut range_eigs_sorted: Vec<f64> = range_eigenvalues_sorted;

    // Shrinkage floor: add a rho-independent ridge to the penalized block eigenvalues.
    // This prevents barely-penalized directions from causing pathological non-Gaussianity
    // in the posterior (extreme skewness under non-canonical links like logit with
    // high-dimensional spatial smooths). The ridge magnitude is proportional to the
    // balanced penalty's max eigenvalue (lambda-independent scale), so LAML gradients
    // w.r.t. rho remain correct: d(epsilon * I)/d(rho_k) = 0.
    let shrinkage_ridge = penalty_shrinkage_floor
        .filter(|&eps| eps > 0.0)
        .map(|eps| eps * invariant.max_balanced_eigenvalue)
        .unwrap_or(0.0);
    if shrinkage_ridge > 0.0 {
        let min_eig_before = range_eigs_sorted
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        for eig in range_eigs_sorted.iter_mut() {
            *eig += shrinkage_ridge;
        }
        // Log when the floor materially changes the smallest eigenvalue (>1% relative shift).
        if min_eig_before > 0.0 && shrinkage_ridge / min_eig_before > 0.01 {
            log::info!(
                "Penalty shrinkage floor active: ridge={:.3e} (min_eig_before={:.3e}, ratio={:.1e}, max_bal_eig={:.3e})",
                shrinkage_ridge,
                min_eig_before,
                shrinkage_ridge / min_eig_before,
                invariant.max_balanced_eigenvalue,
            );
        }
    }

    let max_eig = range_eigs_sorted
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let eigenvalue_floor = max_eig * 1e-12;
    let qs = compose_qs_from_split(&q_pen, &q_null, p);

    // Guard against any accidental penalized/null mixing. The transformed penalty
    // roots must have negligible support on null columns by construction.
    let leakage = assess_subspace_leakage(&qs, &rs_transformed, structural_rank, p);
    let leakage_rel_tol = 1e-10;
    let leakage_abs_tol = 1e-12;
    let orth_tol = 1e-10;
    if leakage.max_rel_sq > leakage_rel_tol && leakage.max_abs_sq > leakage_abs_tol
        || leakage.max_cross_gram_abs > orth_tol
    {
        return Err(EstimationError::LayoutError(format!(
            "Reparameterization subspace split is inconsistent: max null leakage {:.3e} (rel {:.3e}, worst penalty {}), max |Qp'Qn| {:.3e}",
            leakage.max_abs_sq.sqrt(),
            leakage.max_rel_sq.sqrt(),
            leakage.worst_penalty,
            leakage.max_cross_gram_abs,
        )));
    }

    // Truncated basis in transformed coordinates:
    //   U_⊥^(t) = Qs^T U_⊥^(orig) = Qs^T Q_n.
    let mut u_truncated_mat = Mat::<f64>::zeros(p, q_null.ncols());
    matmul(
        u_truncated_mat.as_mut(),
        Accum::Replace,
        qs.transpose(),
        q_null.as_ref(),
        1.0,
        Par::Seq,
    );

    // E is represented in TRANSFORMED coordinates (beta_t).  Because the
    // penalized subspace is NOT rotated by the lambda-dependent eigenvectors
    // (to keep Q_s stable across BFGS iterations), E is no longer diagonal.
    // Instead E = diag(√d) · U' embedded in structural_rank × p, so that
    // E'E = U diag(d) U' = Σ λ_k S_k in the invariant penalized basis.
    let mut e_transformed_mat = Mat::<f64>::zeros(structural_rank, p);
    for row_idx in 0..structural_rank {
        let safe_eigenval = range_eigs_sorted[row_idx].max(eigenvalue_floor);
        let sqrt_eigenval = safe_eigenval.sqrt();
        // E[row, j] = sqrt(d_row) * U'[row, j] = sqrt(d_row) * U[j, row]
        for j in 0..penalized_rank {
            e_transformed_mat[(row_idx, j)] = sqrt_eigenval * range_rotation[(j, row_idx)];
        }
    }

    // Smooth δ-regularized pseudo-logdet: L_δ(S) = log det(S + δI) − m₀ log δ.
    //
    // This replaces the hard ε-threshold truncation of null eigenvalues,
    // making the objective C∞ in outer parameters θ and eliminating
    // artificial kinks when eigenvalues cross the threshold.
    //
    // m₀ is the nullity: p_dim − structural_rank (number of null eigenvalues).
    // δ is chosen proportional to machine epsilon × spectral scale.
    //
    // Reference: response.md Section 7.
    let nullity = penalized_rank.saturating_sub(structural_rank);
    let delta = {
        let max_ev = range_eigs_sorted
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            .max(1.0);
        1e-10 * max_ev
    };
    let log_det: f64 = {
        let log_det_reg: f64 = range_eigs_sorted
            .iter()
            .take(penalized_rank)
            .map(|&ev| (ev + delta).ln())
            .sum();
        log_det_reg - (nullity as f64) * delta.ln()
    };

    let mut det1vec = vec![0.0; lambdas.len()];

    for (k, lambda) in lambdas.iter().enumerate() {
        let s_k = &s_k_transformed_cache[k];
        // Compute tr((S+δI)⁻¹ S_k) via the eigenbasis to avoid precision loss
        // from materializing s_reg_inv.  Each eigencomponent contributes
        //   (U^T S_k U)_{l,l} / (d_l + δ),
        // which we evaluate without forming the full rotated matrix.
        let mut trace = 0.0_f64;
        for l in 0..penalized_rank {
            let eigenval = range_eigs_sorted[l];
            let inv_d = 1.0 / (eigenval + delta);
            // (U^T S_k U)_{l,l} = sum_{i,j} U[i,l] * S_k[i,j] * U[j,l]
            let mut diag_ll = 0.0_f64;
            for i in 0..penalized_rank {
                for j in 0..penalized_rank {
                    diag_ll += range_rotation[(i, l)] * s_k[(i, j)] * range_rotation[(j, l)];
                }
            }
            trace += inv_d * diag_ll;
        }
        det1vec[k] = *lambda * trace;
    }

    #[cfg(debug_assertions)]
    {
        // Algebraic guardrail: cross-check the eigenbasis det1 against the
        // materialized (S+δI)⁻¹ matrix contraction.  The eigenbasis path is
        // primary; this validates that s_reg_inv is consistent.
        let mut s_reg_inv = Mat::<f64>::zeros(p, p);
        for l in 0..penalized_rank {
            let eigenval = range_eigs_sorted[l];
            let inv_d = 1.0 / (eigenval + delta);
            for i in 0..penalized_rank {
                for j in 0..penalized_rank {
                    s_reg_inv[(i, j)] += inv_d * range_rotation[(i, l)] * range_rotation[(j, l)];
                }
            }
        }
        let mut maxdet1_mismatch = 0.0_f64;
        for (k, lambda) in lambdas.iter().enumerate() {
            let s_k = &s_k_transformed_cache[k];
            // Reference: tr(s_reg_inv * S_k) restricted to the penalized block
            // (s_reg_inv is structurally zero outside it).
            let mut trace = 0.0_f64;
            for i in 0..penalized_rank {
                for j in 0..penalized_rank {
                    trace += s_reg_inv[(i, j)] * s_k[(j, i)];
                }
            }
            let reference = *lambda * trace;
            maxdet1_mismatch = maxdet1_mismatch.max((reference - det1vec[k]).abs());
        }
        assert!(
            maxdet1_mismatch <= 1e-9,
            "det1 mismatch between optimized and reference formulas: max_abs={maxdet1_mismatch:.3e}"
        );
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

    #[cfg(debug_assertions)]
    {
        // Structural check: transformed S must not leak into declared null coordinates.
        let mut max_null_diag = 0.0_f64;
        let mut max_null_offdiag = 0.0_f64;
        for i in structural_rank..p {
            max_null_diag = max_null_diag.max(s_truncated[(i, i)].abs());
            for j in 0..p {
                if i != j {
                    max_null_offdiag = max_null_offdiag.max(s_truncated[(i, j)].abs());
                }
            }
        }
        assert!(
            max_null_diag <= 1e-10 && max_null_offdiag <= 1e-10,
            "null-space leakage in transformed penalty: max_null_diag={max_null_diag:.3e}, max_null_offdiag={max_null_offdiag:.3e}"
        );
    }

    Ok(ReparamResult {
        s_transformed: mat_to_array(&s_truncated),
        log_det,
        det1: Array1::from(det1vec),
        qs: mat_to_array(&qs),
        rs_transformed: rs_transformed.iter().map(mat_to_array).collect(),
        rs_transposed: rs_transformed
            .iter()
            .map(|mat| Array2::from_shape_fn((mat.ncols(), mat.nrows()), |(i, j)| mat[(j, i)]))
            .collect(),
        e_transformed: mat_to_array(&e_transformed_mat),
        u_truncated: mat_to_array(&u_truncated_mat),
        penalty_shrinkage_ridge: shrinkage_ridge,
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
///
/// When `cached_invariant` is `Some`, reuses the precomputed eigendecomposition
/// (the hot path inside the REML loop). When `None`, computes the invariant on
/// the fly (the post-REML refit path). Merging both cases into a single entry
/// point ensures `penalty_shrinkage_floor` is always applied regardless of
/// whether a cached invariant is available.
pub fn stable_reparameterization_engine(
    rs_list: &[Array2<f64>],
    lambdas: &[f64],
    dims: EngineDims,
    cached_invariant: Option<&ReparamInvariant>,
    penalty_shrinkage_floor: Option<f64>,
) -> Result<ReparamResult, EstimationError> {
    let owned;
    let invariant = match cached_invariant {
        Some(inv) => inv,
        None => {
            owned = precompute_reparam_invariant(rs_list, dims.p)?;
            &owned
        }
    };
    stable_reparameterizationwith_invariant(
        rs_list,
        lambdas,
        dims.p,
        invariant,
        penalty_shrinkage_floor,
    )
}

// ---------------------------------------------------------------------------
// Kronecker-factored reparameterization for tensor-product smooths
// ---------------------------------------------------------------------------

/// Result of Kronecker-factored reparameterization.
///
/// Exploits the fact that for Kronecker-structured penalties, the joint
/// eigenvector matrix is `U_1 ⊗ ... ⊗ U_d` and the reparameterized design
/// is a rowwise Kronecker of `(B_k U_k)` — all remaining factored.
#[derive(Clone)]
pub struct KroneckerReparamResult {
    /// Reparameterized marginal designs: `B_k · U_k` for each marginal k.
    pub reparameterized_marginals: Vec<Array2<f64>>,
    /// Marginal eigenvalues from each marginal penalty eigendecomposition.
    pub marginal_eigenvalues: Vec<Array1<f64>>,
    /// Marginal eigenvector matrices U_k.
    pub marginal_qs: Vec<Array2<f64>>,
    /// log|S|₊ computed from marginal eigenvalue grid.
    pub log_det: f64,
    /// First derivatives of log|S|₊ w.r.t. ρ_k = log(λ_k).
    pub det1: Array1<f64>,
    /// Second derivatives of log|S|₊ w.r.t. ρ.
    pub det2: Array2<f64>,
    /// Shrinkage ridge added to eigenvalues (if any).
    pub penalty_shrinkage_ridge: f64,
    /// Whether a double penalty (global ridge) is present.
    pub has_double_penalty: bool,
    /// Marginal basis dimensions.
    pub marginal_dims: Vec<usize>,
}

impl KroneckerReparamResult {
    /// Materialize the joint Qs matrix (U_1 ⊗ ... ⊗ U_d) as dense p×p.
    /// Only for fallback paths — avoid in hot loops.
    pub fn materialize_qs(&self) -> Array2<f64> {
        let mut qs = Array2::<f64>::eye(1);
        for u_k in &self.marginal_qs {
            qs = kronecker_product(&qs, u_k);
        }
        qs
    }

    /// Materialize s_transformed (the penalty in the reparameterized basis).
    /// In the eigenbasis, this is diagonal with entries Σ_k λ_k μ_{k,j_k}.
    pub fn materialize_s_transformed(&self, lambdas: &[f64]) -> Array2<f64> {
        let d = self.marginal_dims.len();
        let p: usize = self.marginal_dims.iter().copied().product();
        let mut s = Array2::<f64>::zeros((p, p));

        let mut multi_idx = vec![0usize; d];
        let mut flat = 0usize;
        loop {
            let mut sigma = self.penalty_shrinkage_ridge;
            for k in 0..d {
                sigma += lambdas[k] * self.marginal_eigenvalues[k][multi_idx[k]];
            }
            if self.has_double_penalty && lambdas.len() > d {
                sigma += lambdas[d];
            }
            s[[flat, flat]] = sigma;
            flat += 1;

            let mut carry = true;
            for dim in (0..d).rev() {
                if carry {
                    multi_idx[dim] += 1;
                    if multi_idx[dim] < self.marginal_dims[dim] {
                        carry = false;
                    } else {
                        multi_idx[dim] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
        s
    }

    /// Convert to a standard ReparamResult for compatibility with the existing
    /// solver infrastructure.  This materializes the dense Qs and transformed
    /// penalty — O(p²) — but allows the factored path to integrate without
    /// modifying every downstream consumer.
    pub fn to_standard_reparam_result(
        &self,
        rs_list: &[Array2<f64>],
        lambdas: &[f64],
        p: usize,
    ) -> Result<ReparamResult, EstimationError> {
        let qs = self.materialize_qs();
        let s_transformed = self.materialize_s_transformed(lambdas);

        // Transform penalty roots: R_k_transformed = R_k · Qs
        let rs_transformed: Vec<Array2<f64>> = rs_list
            .iter()
            .map(|r| r.dot(&qs))
            .collect();
        let rs_transposed: Vec<Array2<f64>> = rs_transformed.iter().map(|r| r.t().to_owned()).collect();

        // Build e_transformed: combined penalty square root in transformed coords.
        // For Kronecker structure, the penalty is diagonal in the eigenbasis.
        // e_transformed rows are the nonzero rows of sqrt(Σ_k λ_k S_k)^{1/2}.
        let d = self.marginal_dims.len();
        let diag_vals: Vec<f64> = {
            let mut vals = Vec::with_capacity(p);
            let mut multi_idx = vec![0usize; d];
            loop {
                let mut sigma = self.penalty_shrinkage_ridge;
                for k in 0..d {
                    sigma += lambdas[k] * self.marginal_eigenvalues[k][multi_idx[k]];
                }
                if self.has_double_penalty && lambdas.len() > d {
                    sigma += lambdas[d];
                }
                vals.push(if sigma > 0.0 { sigma.sqrt() } else { 0.0 });

                let mut carry = true;
                for dim in (0..d).rev() {
                    if carry {
                        multi_idx[dim] += 1;
                        if multi_idx[dim] < self.marginal_dims[dim] {
                            carry = false;
                        } else {
                            multi_idx[dim] = 0;
                        }
                    }
                }
                if carry {
                    break;
                }
            }
            vals
        };
        let rank = diag_vals.iter().filter(|&&v| v > 1e-12).count();
        let mut e_transformed = Array2::<f64>::zeros((rank, p));
        let mut row = 0;
        for (j, &v) in diag_vals.iter().enumerate() {
            if v > 1e-12 {
                e_transformed[[row, j]] = v;
                row += 1;
            }
        }

        // u_truncated: null-space eigenvectors (columns with zero eigenvalue).
        let null_count = p - rank;
        let mut u_truncated = Array2::<f64>::zeros((p, null_count));
        let mut col = 0;
        for (j, &v) in diag_vals.iter().enumerate() {
            if v <= 1e-12 {
                u_truncated[[j, col]] = 1.0; // standard basis vector in eigenbasis
                col += 1;
            }
        }

        Ok(ReparamResult {
            s_transformed,
            log_det: self.log_det,
            det1: self.det1.clone(),
            qs,
            rs_transformed,
            rs_transposed,
            e_transformed,
            u_truncated,
            penalty_shrinkage_ridge: self.penalty_shrinkage_ridge,
        })
    }
}

/// Kronecker-factored reparameterization for tensor-product penalties.
///
/// Instead of eigendecomposing the full p×p balanced penalty (O(p³)), this
/// eigendecomposes each marginal penalty separately (O(Σ q_k³)) and computes
/// the joint eigensystem as the Kronecker product of marginal eigensystems.
pub fn kronecker_reparameterization_engine(
    marginal_designs: &[Array2<f64>],
    marginal_penalties: &[Array2<f64>],
    marginal_dims: &[usize],
    lambdas: &[f64],
    has_double_penalty: bool,
    penalty_shrinkage_floor: Option<f64>,
) -> Result<KroneckerReparamResult, EstimationError> {
    use crate::faer_ndarray::FaerEigh;

    let d = marginal_dims.len();
    if marginal_designs.len() != d || marginal_penalties.len() != d {
        return Err(EstimationError::LayoutError(format!(
            "kronecker_reparameterization_engine: dimension mismatch: designs={}, penalties={}, dims={}",
            marginal_designs.len(),
            marginal_penalties.len(),
            d
        )));
    }

    // Eigendecompose each marginal penalty.
    let mut marginal_eigenvalues = Vec::with_capacity(d);
    let mut marginal_qs = Vec::with_capacity(d);
    for (k, s_k) in marginal_penalties.iter().enumerate() {
        let (evals, evecs) = s_k.eigh(Side::Lower).map_err(|e| {
            EstimationError::LayoutError(format!(
                "kronecker_reparameterization_engine: eigendecomp of marginal {k}: {e}"
            ))
        })?;
        marginal_eigenvalues.push(evals);
        marginal_qs.push(evecs);
    }

    // Reparameterized marginals: B_k · U_k.
    let reparameterized_marginals: Vec<Array2<f64>> = marginal_designs
        .iter()
        .zip(marginal_qs.iter())
        .map(|(b_k, u_k)| b_k.dot(u_k))
        .collect();

    // Compute shrinkage ridge from balanced penalty eigenvalue scale.
    let penalty_shrinkage_ridge = if let Some(floor) = penalty_shrinkage_floor {
        // Max balanced eigenvalue: for Kronecker, the balanced penalty's max
        // eigenvalue is the max over multi-indices of Σ_k (1/||S_k||_F) μ_{k,j_k}.
        let mut max_bal = 0.0_f64;
        let mut multi_idx = vec![0usize; d];
        let frob_norms: Vec<f64> = marginal_penalties
            .iter()
            .map(|s| s.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12))
            .collect();
        loop {
            let mut sigma = 0.0;
            for k in 0..d {
                sigma += marginal_eigenvalues[k][multi_idx[k]] / frob_norms[k];
            }
            max_bal = max_bal.max(sigma);

            let mut carry = true;
            for dim in (0..d).rev() {
                if carry {
                    multi_idx[dim] += 1;
                    if multi_idx[dim] < marginal_dims[dim] {
                        carry = false;
                    } else {
                        multi_idx[dim] = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
        floor * max_bal
    } else {
        0.0
    };

    // Compute logdet and derivatives from marginal eigenvalue grid.
    let n_pen = d + if has_double_penalty { 1 } else { 0 };
    let mut log_det = 0.0;
    let mut det1 = Array1::<f64>::zeros(n_pen);
    let mut det2 = Array2::<f64>::zeros((n_pen, n_pen));
    let tol = 1e-12;

    let mut multi_idx = vec![0usize; d];
    loop {
        let mut sigma = penalty_shrinkage_ridge;
        for k in 0..d {
            sigma += lambdas[k] * marginal_eigenvalues[k][multi_idx[k]];
        }
        if has_double_penalty {
            sigma += lambdas[d];
        }

        if sigma > tol {
            log_det += sigma.ln();
            let inv_sigma = 1.0 / sigma;
            let inv_sigma2 = inv_sigma * inv_sigma;

            for k in 0..d {
                let ck = lambdas[k] * marginal_eigenvalues[k][multi_idx[k]];
                det1[k] += ck * inv_sigma;
            }
            if has_double_penalty {
                det1[d] += lambdas[d] * inv_sigma;
            }

            for k in 0..n_pen {
                let ck = if k < d {
                    lambdas[k] * marginal_eigenvalues[k][multi_idx[k]]
                } else {
                    lambdas[d]
                };
                det2[[k, k]] += ck * inv_sigma - ck * ck * inv_sigma2;
                for l in (k + 1)..n_pen {
                    let cl = if l < d {
                        lambdas[l] * marginal_eigenvalues[l][multi_idx[l]]
                    } else {
                        lambdas[d]
                    };
                    let off_diag = -ck * cl * inv_sigma2;
                    det2[[k, l]] += off_diag;
                    det2[[l, k]] += off_diag;
                }
            }
        }

        let mut carry = true;
        for dim in (0..d).rev() {
            if carry {
                multi_idx[dim] += 1;
                if multi_idx[dim] < marginal_dims[dim] {
                    carry = false;
                } else {
                    multi_idx[dim] = 0;
                }
            }
        }
        if carry {
            break;
        }
    }

    Ok(KroneckerReparamResult {
        reparameterized_marginals,
        marginal_eigenvalues,
        marginal_qs,
        log_det,
        det1,
        det2,
        penalty_shrinkage_ridge,
        has_double_penalty,
        marginal_dims: marginal_dims.to_vec(),
    })
}

/// Calculate the 2-norm condition number of a matrix.
///
/// For symmetric matrices (the dominant case for GAM Hessians/penalties),
/// this uses an eigenvalue path and computes:
///   cond_2(A) = max_i |lambda_i| / min_i |lambda_i|
/// which is exactly equal to the singular-value definition for symmetric A.
///
/// For non-symmetric matrices, this falls back to SVD:
///   cond_2(A) = sigma_max / sigma_min
///
/// This preserves semantics while avoiding full SVD in hot paths.
///
/// # Arguments
/// * `matrix` - The matrix to analyze
///
/// # Returns
/// * `Ok(condition_number)` - The condition number (max_sv / min_sv)
/// * `Ok(f64::INFINITY)` - If the matrix is effectively singular (min_sv < 1e-12)
/// * `Err` - If SVD computation fails
pub fn calculate_condition_number(matrix: &Array2<f64>) -> Result<f64, FaerLinalgError> {
    let (rows, cols) = matrix.dim();
    if rows == 0 || cols == 0 {
        return Ok(1.0);
    }

    // Fast path for (near-)symmetric square matrices.
    if rows == cols {
        let mut max_abs = 0.0_f64;
        let mut max_asym = 0.0_f64;
        for i in 0..rows {
            for j in 0..cols {
                max_abs = max_abs.max(matrix[[i, j]].abs());
            }
            for j in 0..i {
                let diff = (matrix[[i, j]] - matrix[[j, i]]).abs();
                if diff > max_asym {
                    max_asym = diff;
                }
            }
        }
        let sym_tol = max_abs.max(1.0) * 1e-12;
        if max_asym <= sym_tol {
            let (evals, _) = matrix.eigh(Side::Lower)?;
            let mut max_abs_eval = 0.0_f64;
            let mut min_abs_eval = f64::INFINITY;
            for &lam in evals.iter() {
                let s = lam.abs();
                max_abs_eval = max_abs_eval.max(s);
                min_abs_eval = min_abs_eval.min(s);
            }
            if min_abs_eval < 1e-12 {
                return Ok(f64::INFINITY);
            }
            return Ok(max_abs_eval / min_abs_eval);
        }
    }

    // General matrix fallback.
    let (_, s, _) = matrix.svd(false, false)?;
    let max_sv = s.iter().fold(0.0_f64, |max, &val| max.max(val));
    let min_sv = s.iter().fold(f64::INFINITY, |min, &val| min.min(val));
    if min_sv < 1e-12 {
        return Ok(f64::INFINITY);
    }
    Ok(max_sv / min_sv)
}

#[cfg(test)]
mod tests {
    use super::{
        SubspaceLeakageMetrics, assess_subspace_leakage, precompute_reparam_invariant,
        stable_reparameterizationwith_invariant,
    };
    use faer::Mat;
    use ndarray::{Array2, array};

    fn metrics_for(
        qs: &Mat<f64>,
        rs: &[Mat<f64>],
        structural_rank: usize,
        p: usize,
    ) -> SubspaceLeakageMetrics {
        assess_subspace_leakage(qs, rs, structural_rank, p)
    }

    #[test]
    fn subspace_leakage_iszero_for_clean_split() {
        let p = 4usize;
        let structural_rank = 2usize;
        let qs = Mat::<f64>::identity(p, p);
        let mut r0 = Mat::<f64>::zeros(2, p);
        r0[(0, 0)] = 1.0;
        r0[(1, 1)] = 2.0;

        let m = metrics_for(&qs, &[r0], structural_rank, p);
        assert!(m.max_abs_sq <= 1e-16);
        assert!(m.max_rel_sq <= 1e-16);
        assert!(m.max_cross_gram_abs <= 1e-16);
    }

    #[test]
    fn subspace_leakage_detects_null_column_energy() {
        let p = 4usize;
        let structural_rank = 2usize;
        let qs = Mat::<f64>::identity(p, p);
        let mut r0 = Mat::<f64>::zeros(1, p);
        r0[(0, 2)] = 3.0;

        let m = metrics_for(&qs, &[r0], structural_rank, p);
        assert!(m.max_abs_sq > 0.0);
        assert!(m.max_rel_sq > 0.99);
    }

    #[test]
    fn subspace_leakage_detects_qp_qn_nonorthogonality() {
        let p = 3usize;
        let structural_rank = 1usize;
        let mut qs = Mat::<f64>::identity(p, p);
        qs[(0, 1)] = 0.2;
        let r0 = Mat::<f64>::zeros(1, p);

        let m = metrics_for(&qs, &[r0], structural_rank, p);
        assert!(m.max_cross_gram_abs > 1e-3);
    }

    #[test]
    fn u_truncated_is_transformed_frame_in_nonzero_case() {
        let p = 3usize;
        let rs_list = vec![array![[1.0, 0.0, 0.0]]];
        let lambdas = vec![2.0];
        let inv = precompute_reparam_invariant(&rs_list, p).expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&rs_list, &lambdas, p, &inv, None)
            .expect("stable reparam");

        let expected = rep.qs.t().dot(&inv.split.q_null);
        let diff = &rep.u_truncated - &expected;
        let max_abs = diff.iter().copied().map(f64::abs).fold(0.0, f64::max);
        assert!(
            max_abs <= 1e-10,
            "u_truncated frame mismatch: max_abs={max_abs}"
        );
    }

    #[test]
    fn u_truncated_is_identitywhen_no_penalties() {
        let p = 4usize;
        let rs_list: Vec<Array2<f64>> = Vec::new();
        let lambdas: Vec<f64> = Vec::new();
        let inv = precompute_reparam_invariant(&rs_list, p).expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&rs_list, &lambdas, p, &inv, None)
            .expect("stable reparam");
        assert_eq!(rep.u_truncated, Array2::<f64>::eye(p));
    }

    #[test]
    fn transformed_penalty_is_diagonal_in_transformed_frame() {
        let p = 3usize;
        let inv_sqrt2 = 2.0_f64.sqrt().recip();
        // Penalize a rotated direction in original space so Qs is non-trivial.
        let rs_list = vec![array![[inv_sqrt2, inv_sqrt2, 0.0]]];
        let lambdas = vec![4.0];
        let inv = precompute_reparam_invariant(&rs_list, p).expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&rs_list, &lambdas, p, &inv, None)
            .expect("stable reparam");

        assert_eq!(rep.e_transformed.nrows(), 1);
        assert!(rep.e_transformed[[0, 0]].abs() > 0.0);
        assert!(rep.e_transformed[[0, 1]].abs() <= 1e-12);
        assert!(rep.e_transformed[[0, 2]].abs() <= 1e-12);
        // Compute expected det1 from the δ-regularized formula rather than
        // comparing against the idealized rank.  Single rank-1 penalty with
        // eigenvalue ||rs||² = 1 in the penalized block.
        let s_k_eig = 1.0_f64; // single eigenvalue of S_k
        let lambda = 4.0_f64;
        let max_ev = (lambda * s_k_eig).max(1.0);
        let delta = 1e-10 * max_ev;
        let expected_det1 = lambda * s_k_eig / (lambda * s_k_eig + delta);
        assert!((rep.det1[0] - expected_det1).abs() <= 1e-12);

        let s = rep.s_transformed;
        let mut max_offdiag = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                if i != j {
                    max_offdiag = max_offdiag.max(s[[i, j]].abs());
                }
            }
        }
        assert!(
            max_offdiag <= 1e-10,
            "transformed penalty should be diagonal, max offdiag={max_offdiag}"
        );
        assert!(s[[1, 1]].abs() <= 1e-10);
        assert!(s[[2, 2]].abs() <= 1e-10);
    }

    #[test]
    fn det1_matches_rank_for_single_full_rank_penalty() {
        let p = 2usize;
        let inv_sqrt2 = 2.0_f64.sqrt().recip();
        // Q^T for a 45-degree rotation.
        let q_t = [[inv_sqrt2, inv_sqrt2], [-inv_sqrt2, inv_sqrt2]];
        // R = diag(3, 1) * Q^T gives S = Q * diag(9, 1) * Q^T.
        let rs = array![
            [3.0 * q_t[0][0], 3.0 * q_t[0][1]],
            [1.0 * q_t[1][0], 1.0 * q_t[1][1]]
        ];
        let rs_list = vec![rs];
        let lambdas = vec![5.0];

        let inv = precompute_reparam_invariant(&rs_list, p).expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&rs_list, &lambdas, p, &inv, None)
            .expect("stable reparam");

        assert_eq!(rep.e_transformed.nrows(), p);
        let det1 = rep.det1[0];
        // Compute expected det1 from the δ-regularized formula:
        //   det1 = lambda * sum_l d_l / (lambda*d_l + δ)
        // where d_l are eigenvalues of S_k and δ = 1e-10 * max(lambda*d_l).
        let s_k_eigs = [9.0_f64, 1.0_f64];
        let lambda = 5.0_f64;
        let max_ev = s_k_eigs
            .iter()
            .map(|&d| lambda * d)
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let delta = 1e-10 * max_ev;
        let expected_det1: f64 = s_k_eigs
            .iter()
            .map(|&d| lambda * d / (lambda * d + delta))
            .sum();
        assert!(
            (det1 - expected_det1).abs() <= 1e-12,
            "expected det1={expected_det1}, got {det1}",
        );

        let s = rep.s_transformed;
        assert!(s[[0, 1]].abs() <= 1e-10);
        assert!(s[[1, 0]].abs() <= 1e-10);
        assert!(s[[0, 0]] > 0.0);
        assert!(s[[1, 1]] > 0.0);
    }
}
