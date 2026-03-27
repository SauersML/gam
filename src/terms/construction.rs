use crate::basis::analyze_penalty_block;
use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerEigh, FaerLinalgError, FaerSvd};
use crate::linalg::utils::KahanSum;
use crate::smooth::PenaltyStructureHint;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par, Side};
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut2, Axis, s};
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

pub(crate) fn array_to_faer(array: &Array2<f64>) -> Mat<f64> {
    let (rows, cols) = array.dim();
    Mat::from_fn(rows, cols, |i, j| array[[i, j]])
}

pub(crate) fn mat_to_array(mat: &Mat<f64>) -> Array2<f64> {
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

fn symmetrize_faer_matrix_in_place(matrix: &mut Mat<f64>) {
    let n = matrix.nrows().min(matrix.ncols());
    for i in 0..n {
        for j in 0..i {
            let avg = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
            matrix[(i, j)] = avg;
            matrix[(j, i)] = avg;
        }
    }
}

#[cfg(debug_assertions)]
fn orthogonal_similarity_transform_faer(
    matrix: &Mat<f64>,
    block_dim: usize,
    orthogonal: &Mat<f64>,
) -> Mat<f64> {
    let matrix_block = matrix.as_ref().submatrix(0, 0, block_dim, block_dim);
    let cols = orthogonal.ncols();
    let mut temp = Mat::<f64>::zeros(block_dim, cols);
    matmul(
        temp.as_mut(),
        Accum::Replace,
        matrix_block,
        orthogonal.as_ref(),
        1.0,
        Par::Seq,
    );
    let mut rotated = Mat::<f64>::zeros(cols, cols);
    matmul(
        rotated.as_mut(),
        Accum::Replace,
        orthogonal.transpose(),
        temp.as_ref(),
        1.0,
        Par::Seq,
    );
    symmetrize_faer_matrix_in_place(&mut rotated);
    rotated
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

pub(crate) fn robust_eigh_faer(
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

fn robust_eigh(
    matrix: &Array2<f64>,
    side: Side,
    context: &str,
) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
    let matrix_faer = array_to_faer(matrix);
    let (eigenvalues, eigenvectors) = robust_eigh_faer(&matrix_faer, side, context)?;
    Ok((Array1::from_vec(eigenvalues), mat_to_array(&eigenvectors)))
}

pub(crate) fn kronecker_marginal_eigensystems(
    marginal_penalties: &[Array2<f64>],
    context: &str,
) -> Result<Vec<(Array1<f64>, Array2<f64>)>, EstimationError> {
    let mut eigensystems = Vec::with_capacity(marginal_penalties.len());
    for (k, penalty) in marginal_penalties.iter().enumerate() {
        eigensystems.push(robust_eigh(
            penalty,
            Side::Lower,
            &format!("{context} marginal {k}"),
        )?);
    }
    Ok(eigensystems)
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
    /// Canonical penalties in the TRANSFORMED coordinate frame.
    /// The single source of truth for penalty roots in the transformed frame.
    /// Downstream consumers use these for block-local `PenaltyCoordinate`
    /// construction, TK correction, and ext-coord paths.
    pub canonical_transformed: Vec<CanonicalPenalty>,
    /// Lambda-dependent penalty square root in TRANSFORMED coordinates (rank x p matrix).
    /// This is used for applying the actual penalty in the least squares solve.
    pub e_transformed: Array2<f64>,
    /// Truncated eigenvectors (p × m where m = p - structural_rank).
    ///
    /// Coordinate frame note:
    /// - This matrix is stored in the TRANSFORMED coefficient frame (post-`Qs`),
    ///   i.e. it is compatible with `canonical_transformed`, `beta_transformed`,
    ///   and transformed Hessians without additional coordinate mapping.
    ///
    /// These vectors span the structural null space used by positive-part
    /// log-determinant conventions.
    pub u_truncated: Array2<f64>,
    /// The rho-independent shrinkage ridge magnitude that was added to each
    /// eigenvalue of the penalized block. Zero means no shrinkage was applied.
    pub penalty_shrinkage_ridge: f64,
}

// ---------------------------------------------------------------------------
// Kronecker factor decomposition primitives
// ---------------------------------------------------------------------------

/// Per-factor decomposition result for Kronecker penalties.
struct KroneckerFactorDecomp {
    root: Array2<f64>,              // rank_j × q_j
    positive_eigenvalues: Vec<f64>, // length = rank_j
    rank: usize,
    dim: usize,
}

/// Eigendecompose each Kronecker factor separately at O(Σ q_j³).
/// Returns per-factor decompositions, or `None` if any factor is zero.
fn decompose_kronecker_factors(
    factors: &[Array2<f64>],
    context: &str,
) -> Result<Option<Vec<KroneckerFactorDecomp>>, EstimationError> {
    let mut decomps = Vec::with_capacity(factors.len());
    for (j, factor) in factors.iter().enumerate() {
        let q_j = factor.nrows();
        if q_j != factor.ncols() {
            return Err(EstimationError::InvalidInput(format!(
                "{context}: Kronecker factor {j} must be square, got {}x{}",
                factor.nrows(),
                factor.ncols()
            )));
        }
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
            decomps.push(KroneckerFactorDecomp {
                root: Array2::eye(q_j),
                positive_eigenvalues: vec![1.0; q_j],
                rank: q_j,
                dim: q_j,
            });
            continue;
        }
        let analysis = analyze_penalty_block(factor).map_err(|err| {
            EstimationError::InvalidInput(format!(
                "{context}: Kronecker factor {j} eigendecomp failed: {err}"
            ))
        })?;
        if analysis.rank == 0 {
            return Ok(None);
        }
        let mut root_j = Array2::zeros((analysis.rank, q_j));
        let mut pos_eigs = Vec::with_capacity(analysis.rank);
        let mut row_idx = 0;
        for (i, &eigenval) in analysis.eigenvalues.iter().enumerate() {
            if eigenval > analysis.tol {
                let sqrt_ev = eigenval.sqrt();
                let evec = analysis.eigenvectors.column(i);
                for (c, &v) in evec.iter().enumerate() {
                    root_j[[row_idx, c]] = sqrt_ev * v;
                }
                pos_eigs.push(eigenval);
                row_idx += 1;
            }
        }
        decomps.push(KroneckerFactorDecomp {
            root: root_j,
            positive_eigenvalues: pos_eigs,
            rank: analysis.rank,
            dim: q_j,
        });
    }
    Ok(Some(decomps))
}

/// Build the block-local Kronecker root from pre-computed factor decompositions.
fn assemble_kronecker_root_local(decomps: &[KroneckerFactorDecomp]) -> Array2<f64> {
    let mut kron_root = decomps[0].root.clone();
    for fr in &decomps[1..] {
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
    kron_root
}

/// Compute eigenvalues of the Kronecker product from per-factor eigenvalues.
fn kronecker_eigenvalues(decomps: &[KroneckerFactorDecomp], block_dim: usize) -> (Vec<f64>, usize) {
    let mut kron_eigs = decomps[0].positive_eigenvalues.clone();
    for fd in &decomps[1..] {
        let mut new_eigs = Vec::with_capacity(kron_eigs.len() * fd.positive_eigenvalues.len());
        for &a in &kron_eigs {
            for &b in &fd.positive_eigenvalues {
                new_eigs.push(a * b);
            }
        }
        kron_eigs = new_eigs;
    }
    let max_ev = kron_eigs.iter().copied().fold(0.0_f64, f64::max);
    let tol = max_ev * 1e-10 * (block_dim as f64);
    let positive: Vec<f64> = kron_eigs.into_iter().filter(|&ev| ev > tol).collect();
    let nullity = block_dim - positive.len();
    (positive, nullity)
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
    /// Construct a dense (full-width) canonical penalty from a `rank x p` root.
    /// Used to wrap reparam-transformed roots for consumers that expect
    /// `&[CanonicalPenalty]`.
    pub fn from_dense_root(root: Array2<f64>, p: usize) -> Self {
        let local = root.t().dot(&root);
        let positive_eigenvalues = Vec::new(); // not needed for TK paths
        Self {
            root,
            col_range: 0..p,
            total_dim: p,
            nullity: 0,
            local,
            positive_eigenvalues,
        }
    }

    /// Embed the block-local root into a full-width `rank × total_dim` matrix.
    /// For dense penalties (col_range = 0..p), returns the root unchanged.
    pub fn full_width_root(&self) -> Array2<f64> {
        if self.col_range.start == 0 && self.col_range.end == self.total_dim {
            return self.root.clone();
        }
        let rank = self.root.nrows();
        let mut full = Array2::<f64>::zeros((rank, self.total_dim));
        full.slice_mut(ndarray::s![.., self.col_range.clone()])
            .assign(&self.root);
        full
    }

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

    /// Return a reference to the cached local penalty matrix.
    /// Shape: `block_dim x block_dim`.
    pub fn local_ref(&self) -> &Array2<f64> {
        &self.local
    }

    /// Return an owned copy of the local penalty matrix.
    /// Prefer `local_ref()` when a reference suffices.
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
        scale
            * rm.iter()
                .zip(self.root.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f64>()
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

    let (local_matrix, col_range, hint) = match spec {
        PenaltySpec::Block {
            local,
            col_range,
            structure_hint,
        } => {
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
            (local.view(), col_range.clone(), structure_hint.as_ref())
        }
        PenaltySpec::Dense(m) => {
            if m.nrows() != p || m.ncols() != p {
                return Err(EstimationError::InvalidInput(format!(
                    "{context}: dense penalty {idx} must be {p}x{p}, got {}x{}",
                    m.nrows(),
                    m.ncols()
                )));
            }
            (m.view(), 0..p, None)
        }
    };

    let block_dim = col_range.len();

    // ── Ridge fast path: closed-form, no eigendecomposition ──
    if let Some(PenaltyStructureHint::Ridge(scale)) = hint {
        if *scale <= 0.0 {
            return Ok(None);
        }
        let sqrt_scale = scale.sqrt();
        let mut root = Array2::zeros((block_dim, block_dim));
        for i in 0..block_dim {
            root[[i, i]] = sqrt_scale;
        }
        // Ridge penalties are diagonal ⟹ already symmetric, but symmetrize
        // for consistency with the generic path.
        let local_owned = local_matrix.to_owned();
        let local_sym = (&local_owned + &local_owned.t()) * 0.5;
        return Ok(Some(CanonicalPenalty {
            root,
            col_range,
            total_dim: p,
            nullity: 0,
            local: local_sym,
            positive_eigenvalues: vec![*scale; block_dim],
        }));
    }

    // ── Kronecker fast path: single per-factor eigendecomposition ──
    if let Some(PenaltyStructureHint::Kronecker(factors)) = hint {
        let decomps =
            match decompose_kronecker_factors(factors, &format!("{context} penalty {idx}"))? {
                None => return Ok(None),
                Some(d) => d,
            };
        let (positive_eigenvalues, nullity) = kronecker_eigenvalues(&decomps, block_dim);
        if positive_eigenvalues.is_empty() {
            return Ok(None);
        }
        let root = assemble_kronecker_root_local(&decomps);
        let local_owned = local_matrix.to_owned();
        let local_sym = (&local_owned + &local_owned.t()) * 0.5;
        return Ok(Some(CanonicalPenalty {
            root,
            col_range,
            total_dim: p,
            nullity,
            local: local_sym,
            positive_eigenvalues,
        }));
    }

    // ── Generic block-local path: eigendecompose at O(block_dim³) ──
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
        const OVERLAPPING_PENALTY_DENSE_FALLBACK_MAX_P: usize = 4096;
        if p > OVERLAPPING_PENALTY_DENSE_FALLBACK_MAX_P {
            return Err(EstimationError::LayoutError(format!(
                "overlapping penalty root would require dense {}x{} eigendecomposition; \
                 large-model dense fallback is disabled. Keep penalties structured or \
                 extend the overlapping-penalty solver path",
                p, p
            )));
        }
        // Fallback: accumulate into p × p and eigendecompose globally.
        let mut s_balanced = Array2::zeros((p, p));
        for cp in penalties {
            if cp.rank() == 0 {
                continue;
            }
            let local = cp.local_ref();
            let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if frob_norm > 1e-12 {
                let r = &cp.col_range;
                s_balanced
                    .slice_mut(s![r.start..r.end, r.start..r.end])
                    .scaled_add(1.0 / frob_norm, local);
            }
        }
        let (eigenvalues, eigenvectors) =
            robust_eigh(&s_balanced, Side::Lower, "balanced penalty matrix")?;
        let max_eig = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));
        let tolerance = if max_eig > 0.0 {
            max_eig * 1e-12
        } else {
            1e-12
        };
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
        root: Array2<f64>, // rank_b × block_dim
    }
    let mut total_rank = 0usize;
    let mut block_roots = Vec::with_capacity(block_groups.len());

    for (&(start, end), cps) in &block_groups {
        let block_dim = end - start;
        let mut s_balanced_local = Array2::zeros((block_dim, block_dim));

        for cp in cps {
            let local = cp.local_ref();
            let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if frob_norm > 1e-12 {
                s_balanced_local.scaled_add(1.0 / frob_norm, local);
            }
        }

        let (eigenvalues, eigenvectors) =
            robust_eigh(&s_balanced_local, Side::Lower, "balanced penalty block")?;
        let max_eig = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));
        let tolerance = if max_eig > 0.0 {
            max_eig * 1e-12
        } else {
            1e-12
        };
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
        eb.slice_mut(s![
            row_offset..(row_offset + rank_b),
            br.col_range.start..br.col_range.end
        ])
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
    /// The balanced eigenvector matrix Q (p x p). Block-local roots are
    /// transformed on-the-fly as `R_block @ Q[start..end, :]` instead of
    /// storing pre-multiplied full-width roots.
    qs_base: Array2<f64>,
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

/// Precompute the lambda-invariant reparameterization structure from canonical penalties.
///
/// Uses block-local roots directly instead of requiring rank x p global roots.
/// Each `CanonicalPenalty` carries its own block-local root and column range,
/// so the balanced sum can be assembled without ever materializing full-size
/// penalty matrices.
pub fn precompute_reparam_invariant_from_canonical(
    penalties: &[CanonicalPenalty],
    p_total: usize,
) -> Result<ReparamInvariant, EstimationError> {
    use std::cmp::Ordering;

    let m = penalties.len();

    if m == 0 {
        return Ok(ReparamInvariant {
            split: SubspaceSplit::identity(p_total),
            qs_base: Array2::eye(p_total),
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
        let local = cp.local_ref();
        let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if frob_norm > 1e-12 {
            has_nonzero = true;
        }
        let key = (cp.col_range.start, cp.col_range.end);
        block_groups
            .entry(key)
            .or_default()
            .push(PenRef { penalty_index: i });
    }

    if !has_nonzero {
        return Ok(ReparamInvariant {
            split: SubspaceSplit::identity(p_total),
            qs_base: Array2::eye(p_total),
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
        let mut s_balanced = Mat::<f64>::zeros(p_total, p_total);
        for cp in penalties {
            if cp.rank() == 0 {
                continue;
            }
            let local = cp.local_ref();
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

        let mut order: Vec<usize> = (0..p_total).collect();
        order.sort_by(|&i, &j| {
            bal_eigenvalues[j]
                .partial_cmp(&bal_eigenvalues[i])
                .unwrap_or(Ordering::Equal)
                .then(i.cmp(&j))
        });

        let mut qs = Mat::<f64>::zeros(p_total, p_total);
        for (col_idx, &idx) in order.iter().enumerate() {
            for row in 0..p_total {
                qs[(row, col_idx)] = bal_eigenvectors[(row, idx)];
            }
        }

        let max_bal = order
            .iter()
            .map(|&idx| bal_eigenvalues[idx].abs())
            .fold(0.0_f64, f64::max);
        let rank_tol = if max_bal > 0.0 {
            max_bal * 1e-12
        } else {
            1e-12
        };
        let penalized_rank = order
            .iter()
            .take_while(|&&idx| bal_eigenvalues[idx] > rank_tol)
            .count();
        let split = SubspaceSplit::from_ordered_qs(&qs, penalized_rank, p_total)?;

        return Ok(ReparamInvariant {
            split,
            qs_base: mat_to_array(&qs),
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
    let mut covered = vec![false; p_total];
    for cp in penalties {
        for j in cp.col_range.clone() {
            covered[j] = true;
        }
    }
    let uncovered_cols: Vec<usize> = (0..p_total).filter(|j| !covered[*j]).collect();

    struct BlockResult {
        col_range: Range<usize>,
        q_pen_local: Array2<f64>,  // block_dim × pen_rank
        q_null_local: Array2<f64>, // block_dim × null_rank
        /// Column offset of this block's penalized directions within global Q_pen.
        pen_col_offset: usize,
        /// Column offset of this block's null directions within global Q_null.
        null_col_offset: usize,
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
            let local = cp.local_ref();
            let frob_norm = local.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if frob_norm > 1e-12 {
                s_balanced_local.scaled_add(1.0 / frob_norm, local);
                block_has_nonzero = true;
            }
        }

        if !block_has_nonzero {
            block_results.push(BlockResult {
                col_range: start..end,
                q_pen_local: Array2::zeros((block_dim, 0)),
                q_null_local: Array2::eye(block_dim),
                pen_col_offset: 0,  // set later
                null_col_offset: 0, // set later
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
        let rank_tol = if max_bal > 0.0 {
            max_bal * 1e-12
        } else {
            1e-12
        };
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

        global_max_bal = global_max_bal.max(max_bal);
        block_results.push(BlockResult {
            col_range: start..end,
            q_pen_local,
            q_null_local,
            pen_col_offset: 0,  // set later
            null_col_offset: 0, // set later
        });
    }

    // Compute column offsets for each block in the global Q_pen / Q_null layout.
    let total_pen_rank: usize = block_results.iter().map(|br| br.q_pen_local.ncols()).sum();
    let total_null: usize = block_results
        .iter()
        .map(|br| br.q_null_local.ncols())
        .sum::<usize>()
        + uncovered_cols.len();
    {
        let mut pen_off = 0usize;
        let mut null_off = 0usize;
        for br in &mut block_results {
            br.pen_col_offset = pen_off;
            br.null_col_offset = null_off;
            pen_off += br.q_pen_local.ncols();
            null_off += br.q_null_local.ncols();
        }
    }

    let mut q_pen = Array2::zeros((p_total, total_pen_rank));
    let mut q_null = Array2::zeros((p_total, total_null));

    for br in &block_results {
        let start = br.col_range.start;
        let bd = br.q_pen_local.nrows();
        let pen_r = br.q_pen_local.ncols();
        let null_r = br.q_null_local.ncols();
        if pen_r > 0 {
            q_pen
                .slice_mut(s![
                    start..(start + bd),
                    br.pen_col_offset..(br.pen_col_offset + pen_r)
                ])
                .assign(&br.q_pen_local);
        }
        if null_r > 0 {
            q_null
                .slice_mut(s![
                    start..(start + bd),
                    br.null_col_offset..(br.null_col_offset + null_r)
                ])
                .assign(&br.q_null_local);
        }
    }
    let mut null_col = block_results
        .iter()
        .map(|br| br.q_null_local.ncols())
        .sum::<usize>();
    for &j in &uncovered_cols {
        q_null[[j, null_col]] = 1.0;
        null_col += 1;
    }

    let split = SubspaceSplit { q_pen, q_null };

    // Store the global Q_s = [Q_pen | Q_null] from the split.
    // Block-local roots are transformed on-the-fly as R_block @ Q[start..end, :]
    // inside the reparam engine, avoiding O(k * rank * p) storage.
    let qs_global = split.compose_qs();

    Ok(ReparamInvariant {
        split,
        qs_base: qs_global,
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
    penalties: &[CanonicalPenalty],
    lambdas: &[f64],
    p: usize,
    invariant: &ReparamInvariant,
    penalty_shrinkage_floor: Option<f64>,
) -> Result<ReparamResult, EstimationError> {
    let m = penalties.len();

    if lambdas.len() != m {
        return Err(EstimationError::ParameterConstraintViolation(format!(
            "Lambda count mismatch: expected {} lambdas for {} penalties, got {}",
            m,
            m,
            lambdas.len()
        )));
    }

    // No separate length check needed — penalties are matched against lambdas above,
    // and the invariant's qs_base is p x p (dimension-checked by the split).

    if m == 0 {
        return Ok(ReparamResult {
            s_transformed: Array2::zeros((p, p)),
            log_det: 0.0,
            det1: Array1::zeros(0),
            qs: Array2::eye(p),
            canonical_transformed: vec![],
            e_transformed: Array2::zeros((0, p)),
            // All modes truncated when no penalties; already in transformed frame.
            u_truncated: Array2::eye(p),
            penalty_shrinkage_ridge: 0.0,
        });
    }

    if !invariant.has_nonzero {
        let qs = invariant.split.compose_qs();
        let u_truncated = qs.t().dot(&invariant.split.q_null);
        // All penalties are zero — canonical_transformed = originals (no rotation needed).
        let canonical_transformed: Vec<CanonicalPenalty> = penalties.to_vec();
        return Ok(ReparamResult {
            s_transformed: Array2::zeros((p, p)),
            log_det: 0.0,
            det1: Array1::zeros(m),
            qs,
            canonical_transformed,
            e_transformed: Array2::zeros((0, p)),
            u_truncated,
            penalty_shrinkage_ridge: 0.0,
        });
    }

    let q_pen = array_to_faer(&invariant.split.q_pen);
    let q_null = array_to_faer(&invariant.split.q_null);
    let qs_base = array_to_faer(&invariant.qs_base);
    // Compute transformed roots on-the-fly: R_k_block @ Q[start..end, :].
    // This avoids storing k full-width (rank × p) matrices in the invariant.
    let rs_transformed: Vec<Mat<f64>> = penalties
        .iter()
        .map(|cp| {
            let r = &cp.col_range;
            let root_faer = array_to_faer(&cp.root);
            let q_block = qs_base.submatrix(r.start, 0, cp.block_dim(), p);
            let mut product = Mat::<f64>::zeros(cp.rank(), p);
            matmul(
                product.as_mut(),
                Accum::Replace,
                root_faer.as_ref(),
                q_block,
                1.0,
                Par::Seq,
            );
            product
        })
        .collect();

    let penalized_rank = invariant.split.rank();

    let mut range_eigenvalues_sorted: Vec<f64> = Vec::new();
    let mut range_rotation = Mat::<f64>::zeros(penalized_rank, penalized_rank);
    if penalized_rank > 0 {
        let mut range_block = Mat::<f64>::zeros(penalized_rank, penalized_rank);
        for (lambda, rs_k) in lambdas.iter().zip(rs_transformed.iter()) {
            let mut s_k = penalty_from_root_faer(rs_k);
            symmetrize_faer_matrix_in_place(&mut s_k);
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

    let mut s_k_penalized_cache: Vec<Mat<f64>> = Vec::with_capacity(m);
    for rs_k in rs_transformed.iter() {
        let mut s_k = penalty_from_root_faer(rs_k);
        symmetrize_faer_matrix_in_place(&mut s_k);
        s_k_penalized_cache.push(s_k);
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
        let s_k = &s_k_penalized_cache[k];
        // Compute tr((S+δI)⁻¹ S_k) via the eigenbasis to avoid precision loss
        // from materializing s_reg_inv. Each eigencomponent contributes
        //   (U^T S_k U)_{l,l} / (d_l + δ).
        let mut trace = KahanSum::default();
        for l in 0..penalized_rank {
            let inv_d = 1.0 / (range_eigs_sorted[l] + delta);
            let mut diag_ll = KahanSum::default();
            for i in 0..penalized_rank {
                let u_i = range_rotation[(i, l)];
                let mut row_dot = KahanSum::default();
                for j in 0..penalized_rank {
                    row_dot.add(s_k[(i, j)] * range_rotation[(j, l)]);
                }
                diag_ll.add(u_i * row_dot.sum());
            }
            trace.add(inv_d * diag_ll.sum());
        }
        det1vec[k] = *lambda * trace.sum();
    }

    #[cfg(debug_assertions)]
    {
        // Guardrail: cross-check the primary Rayleigh-quotient contraction
        // against a full orthogonal similarity transform, while staying in
        // the same numerically stable eigenbasis coordinates.
        let mut maxdet1_mismatch = 0.0_f64;
        let mut det1_scale = 0.0_f64;
        for (k, lambda) in lambdas.iter().enumerate() {
            let s_k_penalized = &s_k_penalized_cache[k];
            let s_k_eigenbasis = orthogonal_similarity_transform_faer(
                s_k_penalized,
                penalized_rank,
                &range_rotation,
            );
            let mut trace = KahanSum::default();
            for l in 0..penalized_rank {
                trace.add(s_k_eigenbasis[(l, l)] / (range_eigs_sorted[l] + delta));
            }
            let reference = *lambda * trace.sum();
            maxdet1_mismatch = maxdet1_mismatch.max((reference - det1vec[k]).abs());
            det1_scale = det1_scale.max(reference.abs()).max(det1vec[k].abs());
        }
        let det1_tolerance = 1e-7 * det1_scale.max(1.0);
        assert!(
            maxdet1_mismatch <= det1_tolerance,
            "det1 mismatch between optimized and reference formulas: max_abs={maxdet1_mismatch:.3e}, tol={det1_tolerance:.3e}"
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

    let rs_transformed_arr: Vec<Array2<f64>> = rs_transformed.iter().map(mat_to_array).collect();
    let canonical_transformed: Vec<CanonicalPenalty> = rs_transformed_arr
        .iter()
        .map(|r| CanonicalPenalty::from_dense_root(r.clone(), p))
        .collect();
    Ok(ReparamResult {
        s_transformed: mat_to_array(&s_truncated),
        log_det,
        det1: Array1::from(det1vec),
        qs: mat_to_array(&qs),
        canonical_transformed,
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
/// Stable reparameterization from block-local canonical penalties.
pub fn stable_reparameterization_engine_canonical(
    penalties: &[CanonicalPenalty],
    lambdas: &[f64],
    dims: EngineDims,
    cached_invariant: Option<&ReparamInvariant>,
    penalty_shrinkage_floor: Option<f64>,
) -> Result<ReparamResult, EstimationError> {
    let owned;
    let invariant = match cached_invariant {
        Some(inv) => inv,
        None => {
            owned = precompute_reparam_invariant_from_canonical(penalties, dims.p)?;
            &owned
        }
    };
    stable_reparameterizationwith_invariant(
        penalties,
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

    /// Explicitly materialize the dense artifact bundle expected by legacy
    /// downstream consumers. This is not part of the native Kronecker solve path.
    pub fn materialize_dense_artifact_result(
        &self,
        rs_list: &[Array2<f64>],
        lambdas: &[f64],
        p: usize,
    ) -> Result<ReparamResult, EstimationError> {
        const KRONECKER_DENSE_COMPAT_FALLBACK_MAX_P: usize = 4096;
        if p > KRONECKER_DENSE_COMPAT_FALLBACK_MAX_P {
            return Err(EstimationError::LayoutError(format!(
                "Kronecker reparameterization would materialize dense {}x{} compatibility tensors; \
                 large-model dense fallback is disabled. Wire the downstream solver to consume \
                 the factored Kronecker result directly",
                p, p
            )));
        }
        let qs = self.materialize_qs();
        let s_transformed = self.materialize_s_transformed(lambdas);

        // Transform penalty roots: R_k_transformed = R_k · Qs
        let rs_transformed: Vec<Array2<f64>> = rs_list.iter().map(|r| r.dot(&qs)).collect();
        // rs_transposed removed — canonical_transformed is the single source of truth.

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

        let canonical_transformed: Vec<CanonicalPenalty> = rs_transformed
            .iter()
            .map(|r| CanonicalPenalty::from_dense_root(r.clone(), p))
            .collect();
        Ok(ReparamResult {
            s_transformed,
            log_det: self.log_det,
            det1: self.det1.clone(),
            qs,
            canonical_transformed,
            e_transformed,
            u_truncated,
            penalty_shrinkage_ridge: self.penalty_shrinkage_ridge,
        })
    }
}

/// Compute `log|S|₊` and its first/second derivatives w.r.t. `ρ_k = log(λ_k)`
/// from factored marginal eigenvalues.
///
/// Shared implementation for `KroneckerPenaltySystem::logdet_and_derivatives`
/// and `kronecker_reparameterization_engine`.  Iterates over the ∏q_j
/// multi-index grid in O(d · ∏q_j) time with no O(p²) storage.
pub fn kronecker_logdet_and_derivatives(
    marginal_eigenvalues: &[ArrayView1<'_, f64>],
    marginal_dims: &[usize],
    lambdas: &[f64],
    has_double_penalty: bool,
    ridge: f64,
) -> (f64, Array1<f64>, Array2<f64>) {
    let d = marginal_dims.len();
    let n_pen = d + if has_double_penalty { 1 } else { 0 };

    let mut logdet = 0.0;
    let mut grad = Array1::<f64>::zeros(n_pen);
    let mut hess = Array2::<f64>::zeros((n_pen, n_pen));
    let tol = 1e-12;

    let mut multi_idx = vec![0usize; d];
    loop {
        let mut sigma = ridge;
        for k in 0..d {
            sigma += lambdas[k] * marginal_eigenvalues[k][multi_idx[k]];
        }
        if has_double_penalty {
            sigma += lambdas[d];
        }

        if sigma > tol {
            logdet += sigma.ln();
            let inv_sigma = 1.0 / sigma;
            let inv_sigma2 = inv_sigma * inv_sigma;

            for k in 0..d {
                let ck = lambdas[k] * marginal_eigenvalues[k][multi_idx[k]];
                grad[k] += ck * inv_sigma;
            }
            if has_double_penalty {
                grad[d] += lambdas[d] * inv_sigma;
            }

            for k in 0..n_pen {
                let ck = if k < d {
                    lambdas[k] * marginal_eigenvalues[k][multi_idx[k]]
                } else {
                    lambdas[d]
                };
                hess[[k, k]] += ck * inv_sigma - ck * ck * inv_sigma2;
                for l in (k + 1)..n_pen {
                    let cl = if l < d {
                        lambdas[l] * marginal_eigenvalues[l][multi_idx[l]]
                    } else {
                        lambdas[d]
                    };
                    let off = -ck * cl * inv_sigma2;
                    hess[[k, l]] += off;
                    hess[[l, k]] += off;
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

    (logdet, grad, hess)
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
    let d = marginal_dims.len();
    if marginal_designs.len() != d || marginal_penalties.len() != d {
        return Err(EstimationError::LayoutError(format!(
            "kronecker_reparameterization_engine: dimension mismatch: designs={}, penalties={}, dims={}",
            marginal_designs.len(),
            marginal_penalties.len(),
            d
        )));
    }

    // Eigendecompose each marginal penalty once through the same robust path
    // used by KroneckerPenaltySystem so every Kronecker caller sees the same
    // eigensystem and pseudo-logdet surface.
    let mut marginal_eigenvalues = Vec::with_capacity(d);
    let mut marginal_qs = Vec::with_capacity(d);
    for (evals, evecs) in
        kronecker_marginal_eigensystems(marginal_penalties, "kronecker_reparameterization_engine")?
    {
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

    let marginal_eigenvalue_views: Vec<_> = marginal_eigenvalues
        .iter()
        .map(|evals| evals.view())
        .collect();
    let (log_det, det1, det2) = kronecker_logdet_and_derivatives(
        &marginal_eigenvalue_views,
        marginal_dims,
        lambdas,
        has_double_penalty,
        penalty_shrinkage_ridge,
    );

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
        CanonicalPenalty, SubspaceLeakageMetrics, assess_subspace_leakage,
        precompute_reparam_invariant_from_canonical, stable_reparameterizationwith_invariant,
    };
    use crate::construction::kronecker_product;
    use crate::linalg::faer_ndarray::FaerEigh;
    use faer::Mat;
    use ndarray::{Array2, array};

    /// Build CanonicalPenalty values from full-width roots for tests.
    fn canonical_from_roots(rs_list: &[Array2<f64>], p: usize) -> Vec<CanonicalPenalty> {
        rs_list
            .iter()
            .map(|r| {
                let local = r.t().dot(r);
                CanonicalPenalty {
                    root: r.clone(),
                    col_range: 0..p,
                    total_dim: p,
                    nullity: 0,
                    local,
                    positive_eigenvalues: Vec::new(),
                }
            })
            .collect()
    }

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
        let canonical = canonical_from_roots(&rs_list, p);
        let lambdas = vec![2.0];
        let inv = precompute_reparam_invariant_from_canonical(&canonical, p)
            .expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&canonical, &lambdas, p, &inv, None)
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
        let canonical: Vec<CanonicalPenalty> = Vec::new();
        let lambdas: Vec<f64> = Vec::new();
        let inv = precompute_reparam_invariant_from_canonical(&canonical, p)
            .expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&canonical, &lambdas, p, &inv, None)
            .expect("stable reparam");
        assert_eq!(rep.u_truncated, Array2::<f64>::eye(p));
    }

    #[test]
    fn transformed_penalty_is_diagonal_in_transformed_frame() {
        let p = 3usize;
        let inv_sqrt2 = 2.0_f64.sqrt().recip();
        // Penalize a rotated direction in original space so Qs is non-trivial.
        let rs_list = vec![array![[inv_sqrt2, inv_sqrt2, 0.0]]];
        let canonical = canonical_from_roots(&rs_list, p);
        let lambdas = vec![4.0];
        let inv = precompute_reparam_invariant_from_canonical(&canonical, p)
            .expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&canonical, &lambdas, p, &inv, None)
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
        let canonical = canonical_from_roots(&rs_list, p);
        let lambdas = vec![5.0];

        let inv = precompute_reparam_invariant_from_canonical(&canonical, p)
            .expect("precompute invariant");
        let rep = stable_reparameterizationwith_invariant(&canonical, &lambdas, p, &inv, None)
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

    #[test]
    fn kronecker_reparam_logdet_matches_dense() {
        // 2D tensor product: q1=3, q2=4.
        // Marginal penalties: second-order difference matrices.
        let q1 = 3;
        let q2 = 4;
        let s1 = {
            let mut s = Array2::<f64>::zeros((q1, q1));
            // D2' D2 for order 2 on 3 points: [[1,-2,1],[-2,4,-2],[1,-2,1]]... simplified
            s[[0, 0]] = 1.0;
            s[[0, 1]] = -1.0;
            s[[1, 0]] = -1.0;
            s[[1, 1]] = 2.0;
            s[[1, 2]] = -1.0;
            s[[2, 1]] = -1.0;
            s[[2, 2]] = 1.0;
            s
        };
        let s2 = {
            let mut s = Array2::<f64>::zeros((q2, q2));
            s[[0, 0]] = 1.0;
            s[[0, 1]] = -1.0;
            s[[1, 0]] = -1.0;
            s[[1, 1]] = 2.0;
            s[[1, 2]] = -1.0;
            s[[2, 1]] = -1.0;
            s[[2, 2]] = 2.0;
            s[[2, 3]] = -1.0;
            s[[3, 2]] = -1.0;
            s[[3, 3]] = 1.0;
            s
        };

        let lambdas = [2.5, 1.3];
        // Build dense Kronecker penalty: λ1 (S1⊗I) + λ2 (I⊗S2).
        let p = q1 * q2;
        let i1 = Array2::<f64>::eye(q1);
        let i2 = Array2::<f64>::eye(q2);
        let pen0 = kronecker_product(&s1, &i2);
        let pen1 = kronecker_product(&i1, &s2);
        let mut s_dense = Array2::<f64>::zeros((p, p));
        s_dense.scaled_add(lambdas[0], &pen0);
        s_dense.scaled_add(lambdas[1], &pen1);

        // Dense eigendecomposition for reference pseudo-logdet.
        let (evals_dense, _): (ndarray::Array1<f64>, ndarray::Array2<f64>) =
            s_dense.eigh(faer::Side::Lower).unwrap();
        let tol = 1e-12;
        let ref_logdet: f64 = evals_dense
            .iter()
            .filter(|&&v: &&f64| v > tol)
            .map(|&v: &f64| v.ln())
            .sum();

        // Kronecker reparameterization engine.
        let marginal_designs = vec![
            Array2::<f64>::eye(q1), // dummy designs
            Array2::<f64>::eye(q2),
        ];
        let marginal_penalties = vec![s1, s2];
        let kron_result = super::kronecker_reparameterization_engine(
            &marginal_designs,
            &marginal_penalties,
            &[q1, q2],
            &lambdas,
            false,
            None,
        )
        .unwrap();

        let diff = (kron_result.log_det - ref_logdet).abs();
        assert!(
            diff < 1e-8,
            "Kronecker logdet {:.10} vs dense {:.10}, diff={:.3e}",
            kron_result.log_det,
            ref_logdet,
            diff,
        );

        // Check derivatives via central FD in rho-space (rho = log lambda).
        let rhos: Vec<f64> = lambdas.iter().map(|&l| l.ln()).collect();
        let eps = 1e-5;
        for k in 0..2 {
            let mut rho_plus = rhos.clone();
            rho_plus[k] += eps;
            let mut rho_minus = rhos.clone();
            rho_minus[k] -= eps;
            let lam_plus: Vec<f64> = rho_plus.iter().map(|&r| r.exp()).collect();
            let lam_minus: Vec<f64> = rho_minus.iter().map(|&r| r.exp()).collect();
            let result_plus = super::kronecker_reparameterization_engine(
                &marginal_designs,
                &marginal_penalties,
                &[q1, q2],
                &lam_plus,
                false,
                None,
            )
            .unwrap();
            let result_minus = super::kronecker_reparameterization_engine(
                &marginal_designs,
                &marginal_penalties,
                &[q1, q2],
                &lam_minus,
                false,
                None,
            )
            .unwrap();
            let fd_deriv = (result_plus.log_det - result_minus.log_det) / (2.0 * eps);
            let analytic_deriv = kron_result.det1[k];
            let rel_err = if analytic_deriv.abs() > 1e-10 {
                (fd_deriv - analytic_deriv).abs() / analytic_deriv.abs()
            } else {
                (fd_deriv - analytic_deriv).abs()
            };
            assert!(
                rel_err < 1e-4,
                "det1[{k}] mismatch: analytic={:.8}, fd={:.8}, rel_err={:.3e}",
                analytic_deriv,
                fd_deriv,
                rel_err,
            );
        }
    }
}
