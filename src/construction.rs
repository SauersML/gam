use crate::basis::{
    self, BasisOptions, Dense, KnotSource, create_basis, create_bspline_basis_nd_with_knots,
};
use crate::estimate::EstimationError;
use crate::faer_ndarray::{FaerEigh, FaerLinalgError, FaerSvd};
use crate::model::{InteractionPenaltyKind, ModelConfig};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par, Side};
use ndarray::Zip;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayViewMut2, Axis, s};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone)]
pub(crate) struct TrainingData {
    pub y: Array1<f64>,
    pub p: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
    pub weights: Array1<f64>,
}

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

fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Computes the row-wise tensor product (Khatri-Rao product) of two matrices.
/// This creates the design matrix columns for tensor product interactions.
/// Each row of the result is the outer product of the corresponding rows from A and B.
fn row_wise_tensor_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n_samples = a.nrows();
    assert_eq!(
        n_samples,
        b.nrows(),
        "Matrices must have same number of rows"
    );

    let a_cols = a.ncols();
    let b_cols = b.ncols();

    // Trivial early return for degenerate shapes
    if n_samples == 0 || a_cols == 0 || b_cols == 0 {
        return Array2::zeros((n_samples, a_cols * b_cols));
    }
    let mut result = Array2::zeros((n_samples, a_cols * b_cols));
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(r, mut row)| {
            let ar = a.row(r);
            let br = b.row(r);
            let mut k = 0;
            for i in 0..a_cols {
                let ai = ar[i];
                for j in 0..b_cols {
                    row[k] = ai * br[j];
                    k += 1;
                }
            }
        });
    result
}

fn binomial(n: usize, k: usize) -> usize {
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut numerator = 1usize;
    let mut denominator = 1usize;
    for i in 0..k {
        numerator *= n - i;
        denominator *= i + 1;
    }
    numerator / denominator
}

fn build_banded_difference_penalty(
    num_basis_functions: usize,
    order: usize,
) -> Result<PenaltyRepresentation, EstimationError> {
    if order == 0 || order >= num_basis_functions {
        return Err(EstimationError::InvalidInput(format!(
            "Invalid difference penalty order {} for {} basis functions",
            order, num_basis_functions
        )));
    }

    let coeffs: Array1<f64> = Array1::from_iter((0..=order).map(|j| {
        let sign = if (order - j).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        sign * binomial(order, j) as f64
    }));
    let num_rows = num_basis_functions - order;
    let band_contribs: Vec<Array1<f64>> = (0..=order)
        .map(|shift| {
            let len = order + 1 - shift;
            let mut contrib = Array1::<f64>::zeros(len);
            let left = coeffs.slice(s![..len]);
            let right = coeffs.slice(s![shift..shift + len]);
            Zip::from(&mut contrib)
                .and(&left)
                .and(&right)
                .for_each(|c, &l, &r| {
                    *c = l * r;
                });
            contrib
        })
        .collect();

    let mut positive_bands: Vec<Array1<f64>> = (0..=order)
        .map(|shift| Array1::<f64>::zeros(num_basis_functions - shift))
        .collect();

    for row in 0..num_rows {
        for shift in 0..=order {
            let contrib = &band_contribs[shift];
            let len = contrib.len();
            let mut slice = positive_bands[shift].slice_mut(s![row..row + len]);
            Zip::from(&mut slice).and(contrib).for_each(|dest, &val| {
                *dest += val;
            });
        }
    }

    let mut bands = Vec::with_capacity(order * 2 + 1);
    let mut offsets = Vec::with_capacity(order * 2 + 1);

    for shift in (1..=order).rev() {
        offsets.push(-(shift as i32));
        bands.push(positive_bands[shift].clone());
    }
    offsets.push(0);
    bands.push(positive_bands[0].clone());
    for shift in 1..=order {
        offsets.push(shift as i32);
        bands.push(positive_bands[shift].clone());
    }

    Ok(PenaltyRepresentation::Banded { bands, offsets })
}

struct DifferencePenalty {
    representation: PenaltyRepresentation,
    dense: OnceLock<Array2<f64>>,
}

impl DifferencePenalty {
    fn new(representation: PenaltyRepresentation) -> Self {
        Self {
            representation,
            dense: OnceLock::new(),
        }
    }

    fn as_dense(&self) -> &Array2<f64> {
        self.dense
            .get_or_init(|| self.representation.to_block_dense())
    }
}

struct DifferencePenaltyCache {
    cache: HashMap<(usize, usize), Arc<DifferencePenalty>>,
}

impl DifferencePenaltyCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    fn get(
        &mut self,
        ncols: usize,
        order: usize,
    ) -> Result<Arc<DifferencePenalty>, EstimationError> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.cache.entry((ncols, order)) {
            let representation = build_banded_difference_penalty(ncols, order)?;
            e.insert(Arc::new(DifferencePenalty::new(representation)));
        }
        Ok(self
            .cache
            .get(&(ncols, order))
            .expect("penalty cache missing freshly inserted key")
            .clone())
    }
}

/// Holds the layout of the design matrix `X` and penalty matrices `S_i`.
#[derive(Clone)]
pub(crate) struct EngineLayout {
    pub intercept_col: usize,
    pub sex_col: Option<usize>,
    pub pgs_main_cols: Range<usize>,
    /// Unpenalized PC null-space columns (aligned with config.pc_configs order)
    pub pc_null_cols: Vec<Range<usize>>,
    pub penalty_map: Vec<PenalizedBlock>,
    /// Direct indices to avoid string lookups in hot paths
    pub pc_main_block_idx: Vec<usize>,
    pub interaction_block_idx: Vec<usize>,
    /// Penalty-map index for the sex×PGS interaction block (if present)
    pub sex_pgs_block_idx: Option<usize>,
    /// Marginal widths used when constructing each interaction block (pgs, pc)
    pub interaction_factor_widths: Vec<(usize, usize)>,
    pub total_coeffs: usize,
    pub num_penalties: usize,
}

/// The semantic type of a penalized term in the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TermType {
    PcMainEffect,
    Interaction,
    SexPgsInteraction,
}

/// Information about a single penalized block of coefficients.
/// For tensor product interactions, a single block may have multiple penalties.
#[derive(Clone, Debug)]
pub struct PenalizedBlock {
    pub term_name: String,
    pub col_range: Range<usize>,
    /// Indices of penalty matrices associated with this block.
    /// For main effects: single element vector.
    /// For tensor product interactions: multiple elements (one per dimension).
    pub penalty_indices: Vec<usize>,
    /// The semantic type of this term.
    pub term_type: TermType,
}

impl EngineLayout {
    /// Minimal external layout for arbitrary design matrices used by the calibrator or adapters.
    /// Sets only the fields required by PIRLS and reparameterization.
    pub fn external(total_coeffs: usize, num_penalties: usize) -> Self {
        EngineLayout {
            intercept_col: 0,
            sex_col: None,
            pgs_main_cols: 0..0,
            pc_null_cols: vec![],
            penalty_map: vec![],
            pc_main_block_idx: vec![],
            interaction_block_idx: vec![],
            sex_pgs_block_idx: None,
            interaction_factor_widths: vec![],
            total_coeffs,
            num_penalties,
        }
    }
    /// Creates a new layout based on the model configuration and basis dimensions.
    /// Enforces strict dimensional consistency across the entire GAM system.
    pub fn new(
        config: &ModelConfig,
        pc_null_basis_ncols: &[usize],
        pc_range_basis_ncols: &[usize],
        sex_main_basis_ncols: usize,
        pgs_main_basis_ncols: usize,
        pc_interaction_basis_ncols: &[usize],
        pgs_interaction_basis_ncols: usize,
    ) -> Result<Self, EstimationError> {
        let mut penalty_map = Vec::new();
        let mut current_col = 0;
        let mut penalty_idx_counter = 0;
        let mut pc_null_cols: Vec<Range<usize>> = Vec::with_capacity(config.pc_configs.len());
        let mut pc_main_block_idx: Vec<usize> = Vec::with_capacity(config.pc_configs.len());
        let mut sex_pgs_block_idx: Option<usize> = None;

        // Calculate total coefficients first to ensure consistency
        // Formula: total_coeffs = 1 (intercept) + p_pgs_main + p_pc_main + p_interactions
        let p_pgs_main = pgs_main_basis_ncols;
        let sex_pgs_interaction_basis_ncols = if sex_main_basis_ncols > 0 {
            pgs_main_basis_ncols
        } else {
            0
        };
        let p_pc_main: usize = pc_null_basis_ncols
            .iter()
            .zip(pc_range_basis_ncols.iter())
            .map(|(n_null, n_range)| n_null + n_range)
            .sum();
        // For tensor product interactions: each PC gets num_pgs_bases * num_pc_bases coefficients
        // Use unconstrained dimensions for interaction calculations
        let p_interactions: usize = pc_interaction_basis_ncols
            .iter()
            .map(|&num_pc_basis| pgs_interaction_basis_ncols * num_pc_basis)
            .sum();
        let calculated_total_coeffs = 1
            + sex_main_basis_ncols
            + p_pgs_main
            + sex_pgs_interaction_basis_ncols
            + p_pc_main
            + p_interactions;

        let intercept_col = current_col;
        current_col += 1;

        let sex_col = if sex_main_basis_ncols > 0 {
            let start = current_col;
            current_col += sex_main_basis_ncols;
            Some(start)
        } else {
            None
        };

        if sex_main_basis_ncols > 1 {
            return Err(EstimationError::LayoutError(
                "sex_main_basis_ncols must be 1 with current layout encoding".into(),
            ));
        }

        // Reserve capacities to avoid reallocations
        // Estimated number of penalized blocks: one per PC main effect, plus one per interaction (if enabled)
        let estimated_penalized_blocks = config.pc_configs.len()
            + if pgs_interaction_basis_ncols > 0 {
                config.pc_configs.len()
            } else {
                0
            }
            + if sex_pgs_interaction_basis_ncols > 0 {
                1
            } else {
                0
            };
        penalty_map.reserve_exact(estimated_penalized_blocks);
        pc_null_cols.reserve_exact(config.pc_configs.len());

        // Main effect for each PC (null-space and range-space treated as separate penalized blocks)
        for (i, (&n_null, &n_range)) in pc_null_basis_ncols
            .iter()
            .zip(pc_range_basis_ncols.iter())
            .enumerate()
        {
            // Null-space (unpenalized)
            let null_range = current_col..current_col + n_null;
            pc_null_cols.push(null_range.clone());
            current_col += n_null;

            // Add a dedicated penalty for the null-space columns (mgcv select=TRUE style)
            // This ensures the likelihood is bounded in all directions under the Logit link.
            if n_null > 0 {
                penalty_map.push(PenalizedBlock {
                    term_name: format!("f({})_null", config.pc_configs[i].name),
                    col_range: null_range.clone(),
                    penalty_indices: vec![penalty_idx_counter],
                    term_type: TermType::PcMainEffect,
                });
                penalty_idx_counter += 1;
            }

            // Range-space (penalized)
            let range = current_col..current_col + n_range;
            penalty_map.push(PenalizedBlock {
                term_name: format!("f({})", config.pc_configs[i].name),
                col_range: range.clone(),
                penalty_indices: vec![penalty_idx_counter], // Each PC main gets unique penalty index
                term_type: TermType::PcMainEffect,
            });
            pc_main_block_idx.push(penalty_map.len() - 1);
            current_col += n_range;
            penalty_idx_counter += 1; // Increment for next penalty
        }

        // Main effect for PGS (non-constant basis terms)
        // The PGS main effect is unpenalized intentionally.
        let pgs_main_cols = current_col..current_col + pgs_main_basis_ncols;
        current_col += pgs_main_basis_ncols; // Still advance the column counter

        // Sex×PGS varying-coefficient interaction (single block with wiggle penalty)
        if sex_pgs_interaction_basis_ncols > 0 {
            let range = current_col..current_col + sex_pgs_interaction_basis_ncols;
            penalty_map.push(PenalizedBlock {
                term_name: "f(PGS,sex)".to_string(),
                col_range: range.clone(),
                penalty_indices: vec![penalty_idx_counter],
                term_type: TermType::SexPgsInteraction,
            });
            sex_pgs_block_idx = Some(penalty_map.len() - 1);
            current_col += sex_pgs_interaction_basis_ncols;
            penalty_idx_counter += 1;
        }

        // Tensor product interaction effects (number of penalties depends on configuration)
        let mut interaction_block_idx: Vec<usize> = Vec::with_capacity(config.pc_configs.len());
        let mut interaction_factor_widths: Vec<(usize, usize)> =
            Vec::with_capacity(config.pc_configs.len());
        if pgs_interaction_basis_ncols > 0 {
            for (i, &num_pc_basis) in pc_interaction_basis_ncols.iter().enumerate() {
                let num_tensor_coeffs = pgs_interaction_basis_ncols * num_pc_basis;
                let range = current_col..current_col + num_tensor_coeffs;

                let penalty_indices = match config.interaction_penalty {
                    InteractionPenaltyKind::Isotropic => {
                        let indices = vec![penalty_idx_counter];
                        penalty_idx_counter += 1;
                        indices
                    }
                    InteractionPenaltyKind::Anisotropic => {
                        let indices = vec![
                            penalty_idx_counter,
                            penalty_idx_counter + 1,
                            penalty_idx_counter + 2,
                        ];
                        penalty_idx_counter += 3;
                        indices
                    }
                };

                penalty_map.push(PenalizedBlock {
                    term_name: format!("f(PGS,{})", config.pc_configs[i].name),
                    col_range: range.clone(),
                    penalty_indices,
                    term_type: TermType::Interaction,
                });

                interaction_block_idx.push(penalty_map.len() - 1);
                interaction_factor_widths.push((pgs_interaction_basis_ncols, num_pc_basis));
                current_col += num_tensor_coeffs;
            }
        }

        // Total number of individual penalties: one per PC main + one per interaction (whitened)
        // _penalty_idx_counter has already been incremented to the correct total

        // Verify that our calculation matches the actual column count
        if current_col != calculated_total_coeffs {
            return Err(EstimationError::LayoutError(format!(
                "EngineLayout dimension calculation error: calculated total_coeffs={calculated_total_coeffs} but actual current_col={current_col}"
            )));
        }

        // Layout invariants: PC index-aligned vectors must match lengths
        if pc_main_block_idx.len() != pc_null_cols.len() {
            return Err(EstimationError::LayoutError(format!(
                "PC layout vectors misaligned: pc_main_block_idx.len()={} vs pc_null_cols.len()={}",
                pc_main_block_idx.len(),
                pc_null_cols.len()
            )));
        }
        if interaction_block_idx.len() != interaction_factor_widths.len() {
            return Err(EstimationError::LayoutError(format!(
                "Interaction layout vectors misaligned: interaction_block_idx.len()={} vs interaction_factor_widths.len()={}",
                interaction_block_idx.len(),
                interaction_factor_widths.len()
            )));
        }

        Ok(EngineLayout {
            intercept_col,
            sex_col,
            pgs_main_cols,
            pc_null_cols,
            penalty_map,
            pc_main_block_idx,
            interaction_block_idx,
            sex_pgs_block_idx,
            interaction_factor_widths,
            total_coeffs: current_col,
            num_penalties: penalty_idx_counter,
        })
    }
}

/// Constructs the design matrix `X` and a list of individual penalty matrices `S_i`.
/// Returns the design matrix, penalty matrices, model layout, sum-to-zero constraints,
/// knot vectors, range transformations, interaction centering means, and interaction
/// orthogonalization maps (Alpha) for pure-interaction construction.
pub(crate) fn build_design_and_penalty_matrices(
    data: &TrainingData,
    config: &ModelConfig,
) -> Result<
    (
        Array2<f64>,                  // design matrix
        Vec<Array2<f64>>,             // penalty matrices (dense)
        EngineLayout,                  // model layout
        HashMap<String, Array2<f64>>, // sum_to_zero_constraints
        HashMap<String, Array1<f64>>, // knot_vectors
        HashMap<String, Array2<f64>>, // range_transforms
        HashMap<String, Array2<f64>>, // pc_null_transforms
        HashMap<String, Array1<f64>>, // interaction_centering_means
        HashMap<String, Array2<f64>>, // interaction_orth_alpha (per interaction block)
        Vec<PenaltyMatrix>,           // structured penalty representations
    ),
    EstimationError,
> {
    // PRE-EMPTIVE CHECK: Calculate the potential number of coefficients BEFORE building any matrices
    let n_samples = data.y.len();

    // Calculate coefficients for each term based on the config
    let pgs_basis_coeffs = config.pgs_basis_config.num_knots + config.pgs_basis_config.degree + 1;
    let intercept_coeffs = 1;
    let pgs_main_coeffs = pgs_basis_coeffs - 1;
    let sex_main_coeffs = 1;

    let mut pc_main_coeffs = 0;
    let mut interaction_coeffs = 0;
    for pc_config in &config.pc_configs {
        let pc_basis_coeffs = pc_config.basis_config.num_knots + pc_config.basis_config.degree + 1;
        pc_main_coeffs += pc_basis_coeffs - 1;
        interaction_coeffs += (pgs_basis_coeffs - 1) * (pc_basis_coeffs - 1);
    }

    let sex_pgs_interaction_coeffs = if sex_main_coeffs > 0 {
        pgs_main_coeffs
    } else {
        0
    };

    let num_coeffs = intercept_coeffs
        + sex_main_coeffs
        + pgs_main_coeffs
        + sex_pgs_interaction_coeffs
        + pc_main_coeffs
        + interaction_coeffs;

    if num_coeffs > n_samples {
        // FAIL FAST before any expensive calculations
        log::error!(
            "Model is severely over-parameterized: {} coefficients for {} samples",
            num_coeffs,
            n_samples
        );
        return Err(EstimationError::ModelOverparameterized {
            num_coeffs,
            num_samples: n_samples,
            intercept_coeffs,
            sex_main_coeffs,
            pgs_main_coeffs,
            pc_main_coeffs,
            sex_pgs_interaction_coeffs,
            interaction_coeffs,
        });
    }

    // Validate PC configuration against available data
    if config.pc_configs.len() > data.pcs.ncols() {
        let pc_names: Vec<String> = config.pc_configs.iter().map(|pc| pc.name.clone()).collect();
        return Err(EstimationError::InvalidInput(format!(
            "Configuration requests {} Principal Components ({}), but the provided data file only contains {} PC columns.",
            config.pc_configs.len(),
            pc_names.join(", "),
            data.pcs.ncols()
        )));
    }

    let n_samples = data.y.len();

    let n_pcs = config.pc_configs.len();

    // Initialize knot vector, sum-to-zero constraint, and range transform storage
    let mut sum_to_zero_constraints = HashMap::new();
    let mut knot_vectors = HashMap::new();
    let mut range_transforms = HashMap::new();
    let mut pc_null_transforms: HashMap<String, Array2<f64>> = HashMap::new();
    let mut pc_null_projectors: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_pcs);
    let mut interaction_centering_means = HashMap::new();
    let mut interaction_orth_alpha: HashMap<String, Array2<f64>> = HashMap::new();
    let mut diff_penalty_cache = DifferencePenaltyCache::new();

    // Reserve capacities to avoid rehashing/reallocation
    // +1 for PGS entries where applicable
    knot_vectors.reserve(n_pcs + 1);
    range_transforms.reserve(n_pcs + 1);
    pc_null_transforms.reserve(n_pcs);
    interaction_centering_means.reserve(n_pcs + 1);
    interaction_orth_alpha.reserve(n_pcs + 1);

    // Stage: Generate the PGS basis and apply the sum-to-zero constraint
    let (pgs_basis_unc, pgs_knots) = create_basis::<Dense>(
        data.p.view(),
        KnotSource::Generate {
            data_range: config.pgs_range,
            num_internal_knots: config.pgs_basis_config.num_knots,
        },
        config.pgs_basis_config.degree,
        BasisOptions::value(),
    )?;

    // Save PGS knot vector
    knot_vectors.insert("pgs".to_string(), pgs_knots);

    // Build PGS penalty on FULL basis (not arbitrarily sliced)

    // For PGS main effects: use non-intercept columns and apply sum-to-zero constraint
    let pgs_main_basis_unc = pgs_basis_unc.slice(s![.., 1..]).to_owned();
    let (pgs_main_basis, pgs_z_transform) =
        basis::apply_sum_to_zero_constraint(pgs_main_basis_unc.view(), Some(data.weights.view()))?;

    // Create whitened range transform for PGS (used if switching to whitened interactions)
    let s_pgs_main = diff_penalty_cache.get(pgs_main_basis_unc.ncols(), config.penalty_order)?;
    let (z_null_pgs, z_range_pgs) = basis::null_range_whiten(s_pgs_main.as_dense())?;
    let pgs_null_projector = z_null_pgs.dot(&z_null_pgs.t());

    let pgs_isotropic_interaction_basis = if matches!(
        config.interaction_penalty,
        InteractionPenaltyKind::Isotropic
    ) {
        Some(pgs_main_basis_unc.dot(&z_range_pgs))
    } else {
        None
    };

    // Store PGS range transformation for interactions and potential future penalized PGS
    range_transforms.insert("pgs".to_string(), z_range_pgs.clone());

    // Save the PGS sum-to-zero constraint transformation
    sum_to_zero_constraints.insert("pgs_main".to_string(), pgs_z_transform.clone());

    // Stage: Generate range-only bases for PCs (functional ANOVA decomposition)
    let mut pc_range_bases: Vec<Array2<f64>> = Vec::with_capacity(n_pcs);
    let mut pc_null_bases: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_pcs);
    let mut pc_unconstrained_bases_main: Vec<Array2<f64>> = Vec::with_capacity(n_pcs);

    // Check if we have any PCs to process
    if config.pc_configs.is_empty() {
        log::info!("No PCs provided; building PGS-only model.");
    }

    for i in 0..config.pc_configs.len() {
        let pc_col = data.pcs.column(i);
        let pc_config = &config.pc_configs[i];
        let pc_name = &pc_config.name;
        let (pc_basis_unc, pc_knots) = create_basis::<Dense>(
            pc_col.view(),
            KnotSource::Generate {
                data_range: pc_config.range,
                num_internal_knots: pc_config.basis_config.num_knots,
            },
            pc_config.basis_config.degree,
            BasisOptions::value(),
        )?;

        // Save PC knot vector
        knot_vectors.insert(pc_name.clone(), pc_knots);

        // For PC main effects: use non-intercept columns for whitened range basis
        let pc_main_basis_unc = pc_basis_unc.slice(s![.., 1..]);
        pc_unconstrained_bases_main.push(pc_main_basis_unc.to_owned());

        // Create whitened range transform for PC main effects
        let s_pc_main = diff_penalty_cache.get(pc_main_basis_unc.ncols(), config.penalty_order)?;
        let (z_null_pc, z_range_pc) = basis::null_range_whiten(s_pc_main.as_dense())?;

        // PC main effect uses ONLY the range (penalized) part
        let pc_range_basis = pc_main_basis_unc.dot(&z_range_pc);
        pc_range_bases.push(pc_range_basis);

        // Build null-space (if any)
        if z_null_pc.ncols() > 0 {
            let pc_null_basis = pc_main_basis_unc.dot(&z_null_pc);
            pc_null_bases.push(Some(pc_null_basis));
            pc_null_transforms.insert(pc_name.clone(), z_null_pc.clone());
            pc_null_projectors.push(Some(z_null_pc.dot(&z_null_pc.t())));
        } else {
            pc_null_bases.push(None);
            pc_null_projectors.push(None);
        }

        // Store PC range transformation for interactions and main effects
        range_transforms.insert(pc_name.clone(), z_range_pc);
    }

    // Stage: Calculate the layout first to determine matrix dimensions
    let pc_range_ncols: Vec<usize> = pc_range_bases.iter().map(|b| b.ncols()).collect();
    let pc_null_ncols: Vec<usize> = pc_null_bases
        .iter()
        .map(|opt| opt.as_ref().map_or(0, |b| b.ncols()))
        .collect();
    let pgs_range_ncols = range_transforms
        .get("pgs")
        .map(|rt| rt.ncols())
        .unwrap_or(0);
    let pc_range_interaction_ncols: Vec<usize> = config
        .pc_configs
        .iter()
        .map(|pc| {
            range_transforms
                .get(&pc.name)
                .map(|rt| rt.ncols())
                .unwrap_or(0)
        })
        .collect();

    let (pgs_int_ncols, pc_int_ncols): (usize, Vec<usize>) = match config.interaction_penalty {
        InteractionPenaltyKind::Isotropic => {
            let pc_int_ncols = pc_range_interaction_ncols.clone();
            (pgs_range_ncols, pc_int_ncols)
        }
        InteractionPenaltyKind::Anisotropic => {
            let pgs_cols = pgs_main_basis_unc.ncols();
            let pc_cols = pc_unconstrained_bases_main
                .iter()
                .map(|b| b.ncols())
                .collect();
            (pgs_cols, pc_cols)
        }
    };

    if matches!(
        config.interaction_penalty,
        InteractionPenaltyKind::Anisotropic
    ) {
        assert_eq!(pgs_int_ncols, pgs_main_basis_unc.ncols());
        for (idx, basis) in pc_unconstrained_bases_main.iter().enumerate() {
            assert_eq!(pc_int_ncols[idx], basis.ncols());
        }
    }

    let sex_main_ncols = 1;

    let layout = EngineLayout::new(
        config,
        &pc_null_ncols,
        &pc_range_ncols,
        sex_main_ncols,
        pgs_main_basis.ncols(),
        &pc_int_ncols,
        pgs_int_ncols,
    )?;

    // Stage: Create individual penalty matrices—one per PC main and (for anisotropic) three per interaction
    // Each penalty gets its own lambda parameter for optimal smoothing control
    let p = layout.total_coeffs;
    let mut s_list: Vec<Option<PenaltyMatrix>> = vec![None; layout.num_penalties];

    // Precompute the wiggle penalty component for the sex×PGS interaction
    let s_sex_pgs_wiggle = pgs_z_transform
        .t()
        .dot(&(s_pgs_main.as_dense().dot(&pgs_z_transform)));

    // Fill in identity penalties for each penalized block individually
    for block in &layout.penalty_map {
        if block.term_type != TermType::PcMainEffect {
            continue;
        }
        let col_range = block.col_range.clone();
        // Main effects have single penalty index. Normalize identity penalty
        // so each block has unit Frobenius norm regardless of size.
        let penalty_idx = block.penalty_indices[0];
        let m = col_range.len() as f64;
        let alpha = if m > 0.0 { 1.0 / m.sqrt() } else { 1.0 };
        let band = Array1::from_elem(col_range.len(), alpha);
        s_list[penalty_idx] = Some(PenaltyMatrix {
            col_range: col_range.clone(),
            representation: PenaltyRepresentation::Banded {
                bands: vec![band],
                offsets: vec![0],
            },
        });
    }

    if let Some(block_idx) = layout.sex_pgs_block_idx {
        let block = &layout.penalty_map[block_idx];
        let col_range = block.col_range.clone();
        let frob = |m: &Array2<f64>| m.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-12);
        let penalty_idx_wiggle = block.penalty_indices[0];
        let normalized = &s_sex_pgs_wiggle / frob(&s_sex_pgs_wiggle);
        s_list[penalty_idx_wiggle] = Some(PenaltyMatrix {
            col_range: col_range.clone(),
            representation: PenaltyRepresentation::Dense(normalized),
        });
    }

    let s_pgs_interaction = if matches!(
        config.interaction_penalty,
        InteractionPenaltyKind::Anisotropic
    ) && pgs_int_ncols > 0
    {
        Some(diff_penalty_cache.get(pgs_int_ncols, config.penalty_order)?)
    } else {
        None
    };
    let i_pgs_interaction = if matches!(
        config.interaction_penalty,
        InteractionPenaltyKind::Anisotropic
    ) && pgs_int_ncols > 0
    {
        Some(Array2::<f64>::eye(pgs_int_ncols))
    } else {
        None
    };

    for (pc_idx, _) in config.pc_configs.iter().enumerate() {
        if pc_idx >= layout.interaction_block_idx.len() {
            break;
        }
        let block = &layout.penalty_map[layout.interaction_block_idx[pc_idx]];
        let col_range = block.col_range.clone();

        match config.interaction_penalty {
            InteractionPenaltyKind::Isotropic => {
                assert_eq!(block.penalty_indices.len(), 1);
                let penalty_idx = block.penalty_indices[0];
                let m = col_range.len() as f64;
                let alpha = if m > 0.0 { 1.0 / m.sqrt() } else { 1.0 };
                let band = Array1::from_elem(col_range.len(), alpha);
                s_list[penalty_idx] = Some(PenaltyMatrix {
                    col_range: col_range.clone(),
                    representation: PenaltyRepresentation::Banded {
                        bands: vec![band],
                        offsets: vec![0],
                    },
                });
            }
            InteractionPenaltyKind::Anisotropic => {
                // Defer penalty construction until AFTER orthogonalization (alpha) is computed.
                // This allows us to project the penalties into the constrained space
                // and scale them correctly as recommended by Wood (2017).
            }
        }
    }

    // Range transformations will be returned directly to the caller

    // Stage: Assemble the full design matrix `X` using the layout as the guide
    // Following a strict canonical order to match the coefficient flattening logic in model.rs
    let mut x_matrix = Array2::zeros((n_samples, layout.total_coeffs));

    // Stage: Populate the intercept column first
    x_matrix.column_mut(layout.intercept_col).fill(1.0);

    // Stage: Add sex main effect (single column, unpenalized)
    if let Some(sex_col) = layout.sex_col {
        if data.sex.len() != n_samples {
            return Err(EstimationError::LayoutError(format!(
                "Sex vector length {} does not match number of samples {}",
                data.sex.len(),
                n_samples
            )));
        }
        x_matrix.column_mut(sex_col).assign(&data.sex);
    }

    // Stage: Fill main PC effects (null-space first, then penalized range)
    for (pc_idx, pc_config) in config.pc_configs.iter().enumerate() {
        let pc_name = &pc_config.name;
        // Fill null-space columns
        let null_cols = &layout.pc_null_cols[pc_idx];
        if !null_cols.is_empty()
            && let Some(null_basis) = &pc_null_bases[pc_idx]
        {
            if null_basis.nrows() != n_samples {
                return Err(EstimationError::LayoutError(format!(
                    "PC null basis {} has {} rows but expected {} samples",
                    pc_name,
                    null_basis.nrows(),
                    n_samples
                )));
            }
            if null_basis.ncols() != null_cols.len() {
                return Err(EstimationError::LayoutError(format!(
                    "PC null basis {} has {} columns but layout expects {} columns",
                    pc_name,
                    null_basis.ncols(),
                    null_cols.len()
                )));
            }
            x_matrix
                .slice_mut(s![.., null_cols.clone()])
                .assign(null_basis);
        }
        // Fill penalized range-space columns
        for block in &layout.penalty_map {
            if block.term_name == format!("f({pc_name})") {
                let col_range = block.col_range.clone();
                let pc_basis = &pc_range_bases[pc_idx];

                if pc_basis.nrows() != n_samples {
                    return Err(EstimationError::LayoutError(format!(
                        "PC range basis {} has {} rows but expected {} samples",
                        pc_name,
                        pc_basis.nrows(),
                        n_samples
                    )));
                }
                if pc_basis.ncols() != col_range.len() {
                    return Err(EstimationError::LayoutError(format!(
                        "PC range basis {} has {} columns but layout expects {} columns",
                        pc_name,
                        pc_basis.ncols(),
                        col_range.len()
                    )));
                }
                x_matrix.slice_mut(s![.., col_range]).assign(pc_basis);
                break;
            }
        }
    }

    // Stage: Populate the main PGS effect directly from the layout range
    let pgs_range = layout.pgs_main_cols.clone();

    // Validate dimensions before assignment
    if pgs_main_basis.nrows() != n_samples {
        return Err(EstimationError::LayoutError(format!(
            "PGS main basis has {} rows but expected {} samples",
            pgs_main_basis.nrows(),
            n_samples
        )));
    }
    if pgs_main_basis.ncols() != pgs_range.len() {
        return Err(EstimationError::LayoutError(format!(
            "PGS main basis has {} columns but layout expects {} columns",
            pgs_main_basis.ncols(),
            pgs_range.len()
        )));
    }

    x_matrix
        .slice_mut(s![.., pgs_range])
        .assign(&pgs_main_basis);

    // Precompute sqrt(weights) once for interaction orthogonalization
    let w_sqrt = data.weights.mapv(f64::sqrt);
    let w_col = w_sqrt.view().insert_axis(Axis(1));

    // Stage: Add sex×PGS varying-coefficient interaction
    if let Some(block_idx) = layout.sex_pgs_block_idx {
        let block = &layout.penalty_map[block_idx];
        let col_range = block.col_range.clone();
        if pgs_main_basis.ncols() != col_range.len() {
            return Err(EstimationError::LayoutError(format!(
                "Sex×PGS interaction expects {} columns but layout provides {}",
                pgs_main_basis.ncols(),
                col_range.len()
            )));
        }

        let mut sex_pgs_basis = pgs_main_basis.to_owned();
        for (mut row, &sex_value) in sex_pgs_basis.axis_iter_mut(Axis(0)).zip(data.sex.iter()) {
            row *= sex_value;
        }

        let intercept = x_matrix.column(layout.intercept_col).to_owned();
        let sex_main = layout.sex_col.ok_or_else(|| {
            EstimationError::LayoutError(
                "Layout is missing the sex main-effect column required for the sex×PGS interaction"
                    .to_string(),
            )
        })?;
        let sex_main_cols = x_matrix.slice(s![.., sex_main..sex_main + 1]).to_owned();
        let pgs_main = pgs_main_basis.to_owned();

        let m_cols = 1 + sex_main_cols.ncols() + pgs_main.ncols();
        let mut m_matrix = Array2::<f64>::zeros((n_samples, m_cols));
        let mut offset = 0;
        m_matrix.column_mut(offset).assign(&intercept);
        offset += 1;
        m_matrix
            .slice_mut(s![.., offset..offset + sex_main_cols.ncols()])
            .assign(&sex_main_cols);
        offset += sex_main_cols.ncols();
        m_matrix
            .slice_mut(s![.., offset..offset + pgs_main.ncols()])
            .assign(&pgs_main);

        let mw = &m_matrix * &w_col;
        let tw = &sex_pgs_basis * &w_col;
        let gram = mw.t().dot(&mw);
        let rhs = mw.t().dot(&tw);
        let (u_opt, s, vt_opt) = gram
            .svd(true, true)
            .map_err(EstimationError::EigendecompositionFailed)?;
        let (u, vt) = match (u_opt, vt_opt) {
            (Some(u), Some(vt)) => (u, vt),
            _ => {
                return Err(EstimationError::LayoutError(
                    "SVD did not return U/VT for sex×PGS orthogonalization".to_string(),
                ));
            }
        };
        let smax = s.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let tol = smax * 1e-12;
        let mut s_inv = Array2::zeros((s.len(), s.len()));
        for i in 0..s.len() {
            if s[i] > tol {
                s_inv[[i, i]] = 1.0 / s[i];
            }
        }
        let gram_pinv = vt.t().dot(&s_inv).dot(&u.t());
        let alpha = gram_pinv.dot(&rhs);
        let sex_pgs_orth = &sex_pgs_basis - &m_matrix.dot(&alpha);

        let interaction_key = "f(PGS,sex)".to_string();
        interaction_orth_alpha.insert(interaction_key.clone(), alpha);
        interaction_centering_means
            .insert(interaction_key.clone(), Array1::zeros(sex_pgs_orth.ncols()));
        x_matrix.slice_mut(s![.., col_range]).assign(&sex_pgs_orth);
    }

    // Stage: Populate tensor product interaction effects.
    if !layout.interaction_block_idx.is_empty() {
        for (pc_idx, pc_config) in config.pc_configs.iter().enumerate() {
            if pc_idx >= layout.interaction_block_idx.len() {
                break;
            }
            let pc_name = &pc_config.name;
            // Directly index the corresponding tensor product block
            let tensor_block = &layout.penalty_map[layout.interaction_block_idx[pc_idx]];

            let tensor_interaction = match config.interaction_penalty {
                InteractionPenaltyKind::Isotropic => {
                    let pgs_int_basis =
                        pgs_isotropic_interaction_basis.as_ref().ok_or_else(|| {
                            EstimationError::LayoutError(
                                "PGS interaction basis missing in isotropic mode".to_string(),
                            )
                        })?;
                    let pc_int_basis = &pc_range_bases[pc_idx];
                    row_wise_tensor_product(pgs_int_basis, pc_int_basis)
                }
                InteractionPenaltyKind::Anisotropic => {
                    let pgs_knots = knot_vectors.get("pgs").ok_or_else(|| {
                        EstimationError::LayoutError(
                            "Missing PGS knot vector for tensor interaction".to_string(),
                        )
                    })?;
                    let pc_knots = knot_vectors.get(pc_name).ok_or_else(|| {
                        EstimationError::LayoutError(format!(
                            "Missing knot vector for tensor interaction PC {pc_name}",
                        ))
                    })?;
                    let (tensor_full_arc, _) = create_bspline_basis_nd_with_knots(
                        &[data.p.view(), data.pcs.column(pc_idx).view()],
                        &[pgs_knots.view(), pc_knots.view()],
                        &[
                            config.pgs_basis_config.degree,
                            pc_config.basis_config.degree,
                        ],
                    )?;
                    let tensor_full = (*tensor_full_arc).clone();
                    let k_pgs = pgs_knots.len() - config.pgs_basis_config.degree - 1;
                    let k_pc = pc_knots.len() - pc_config.basis_config.degree - 1;
                    if k_pgs < 2 || k_pc < 2 {
                        return Err(EstimationError::LayoutError(format!(
                            "Tensor interaction requires at least 2 basis functions per dimension (pgs={k_pgs}, pc={k_pc})",
                        )));
                    }
                    let mut keep_cols = Vec::with_capacity((k_pgs - 1) * (k_pc - 1));
                    for i in 1..k_pgs {
                        for j in 1..k_pc {
                            keep_cols.push(i * k_pc + j);
                        }
                    }
                    tensor_full.select(Axis(1), &keep_cols)
                }
            };

            // Validate dimensions
            let col_range = tensor_block.col_range.clone();
            if tensor_interaction.nrows() != n_samples {
                return Err(EstimationError::LayoutError(format!(
                    "Range×Range tensor interaction f(PGS,{}) has {} rows but expected {} samples",
                    pc_name,
                    tensor_interaction.nrows(),
                    n_samples
                )));
            }
            if tensor_interaction.ncols() != col_range.len() {
                return Err(EstimationError::LayoutError(format!(
                    "Range×Range tensor interaction f(PGS,{}) has {} columns but layout expects {} columns",
                    pc_name,
                    tensor_interaction.ncols(),
                    col_range.len()
                )));
            }

            // Build main-effect matrix M = [Intercept | Sex | PGS_main | PC_main_for_this_pc (null + range)]
            // Extract intercept, sex, and PGS main
            let intercept = x_matrix.column(layout.intercept_col).to_owned();
            let sex_main = layout
                .sex_col
                .map(|col| x_matrix.slice(s![.., col..col + 1]).to_owned());
            let pgs_main = x_matrix
                .slice(s![.., layout.pgs_main_cols.clone()])
                .to_owned();
            // Extract this PC's main effect columns: null then range
            let pc_block = &layout.penalty_map[layout.pc_main_block_idx[pc_idx]];
            let pc_null_cols = &layout.pc_null_cols[pc_idx];
            let pc_null = if !pc_null_cols.is_empty() {
                Some(x_matrix.slice(s![.., pc_null_cols.clone()]).to_owned())
            } else {
                None
            };
            let pc_range = x_matrix
                .slice(s![.., pc_block.col_range.clone()])
                .to_owned();

            // Preallocate M = [Intercept | Sex (opt) | PGS_main | PC_null (opt) | PC_range]
            let m_cols = 1
                + sex_main.as_ref().map_or(0, |z| z.ncols())
                + pgs_main.ncols()
                + pc_null.as_ref().map_or(0, |z| z.ncols())
                + pc_range.ncols();
            let mut m_matrix = Array2::<f64>::zeros((n_samples, m_cols));
            let mut offset = 0;
            // Intercept
            m_matrix.column_mut(offset).assign(&intercept);
            offset += 1;
            // Sex main effect (if present)
            if let Some(sex_col) = sex_main.as_ref() {
                m_matrix
                    .slice_mut(s![.., offset..offset + sex_col.ncols()])
                    .assign(sex_col);
                offset += sex_col.ncols();
            }
            // PGS_main
            m_matrix
                .slice_mut(s![.., offset..offset + pgs_main.ncols()])
                .assign(&pgs_main);
            offset += pgs_main.ncols();
            // PC_null
            if let Some(pc_n) = pc_null.as_ref() {
                m_matrix
                    .slice_mut(s![.., offset..offset + pc_n.ncols()])
                    .assign(pc_n);
                offset += pc_n.ncols();
            }
            // PC_range
            m_matrix
                .slice_mut(s![.., offset..offset + pc_range.ncols()])
                .assign(&pc_range);

            // Weighted projection: Alpha = (M^T W M)^+ (M^T W T)
            // Reuse precomputed weight column once
            let mw = &m_matrix * &w_col;
            let tw = &tensor_interaction * &w_col;

            // Compute Gram matrix and RHS
            let gram = mw.t().dot(&mw);
            let rhs = mw.t().dot(&tw);

            // Pseudo-inverse via SVD
            let (u_opt, s, vt_opt) = gram
                .svd(true, true)
                .map_err(EstimationError::EigendecompositionFailed)?;
            let (u, vt) = match (u_opt, vt_opt) {
                (Some(u), Some(vt)) => (u, vt),
                _ => {
                    return Err(EstimationError::LayoutError(
                        "SVD did not return U/VT for interaction orthogonalization".to_string(),
                    ));
                }
            };
            let smax = s.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let tol = smax * 1e-12;
            // Construct Σ^+ as diagonal with thresholding
            let mut s_inv = Array2::zeros((s.len(), s.len()));
            for i in 0..s.len() {
                if s[i] > tol {
                    s_inv[[i, i]] = 1.0 / s[i];
                }
            }
            let gram_pinv = vt.t().dot(&s_inv).dot(&u.t());
            let alpha = gram_pinv.dot(&rhs); // shape: m_cols x t_cols

            // Orthogonalize tensor columns to the space spanned by M (pure interaction)
            let tensor_orth = &tensor_interaction - &m_matrix.dot(&alpha);

            // Save Alpha for use at prediction time
            let interaction_key = format!("f(PGS,{})", pc_name);
            interaction_orth_alpha.insert(interaction_key.clone(), alpha.clone());

            // Store zero means for legacy-free behavior
            let zeros = Array1::zeros(tensor_orth.ncols());
            interaction_centering_means.insert(interaction_key.clone(), zeros);

            // Assign the orthogonalized tensor block to design matrix (no extra centering)
            x_matrix
                .slice_mut(s![.., col_range.clone()])
                .assign(&tensor_orth);

            // Anisotropic penalty construction (projected)
            // This is done HERE because we need `alpha` to project the penalty matrices
            if let InteractionPenaltyKind::Anisotropic = config.interaction_penalty {
                // Ensure layout validity
                if tensor_block.penalty_indices.len() != 3 {
                    return Err(EstimationError::LayoutError(format!(
                        "Interaction block {} should have exactly 3 penalty indices in anisotropic mode, found {}",
                        tensor_block.term_name,
                        tensor_block.penalty_indices.len()
                    )));
                }

                let pgs_cols = pgs_int_ncols;
                let pc_cols = pc_int_ncols[pc_idx];
                if col_range.len() != pgs_cols * pc_cols {
                    return Err(EstimationError::LayoutError(format!(
                        "Interaction block f(PGS,{}) expects {} columns ({}×{}), but layout provided {}",
                        pc_name,
                        pgs_cols * pc_cols,
                        pgs_cols,
                        pc_cols,
                        col_range.len()
                    )));
                }

                // Retrieve raw marginal penalties
                let s_pgs_penalty = s_pgs_interaction
                    .as_ref()
                    .expect("PGS penalty matrix missing in anisotropic mode");
                let i_pgs = i_pgs_interaction
                    .as_ref()
                    .expect("PGS identity matrix missing in anisotropic mode");
                let s_pc_penalty = diff_penalty_cache.get(pc_cols, config.penalty_order)?;
                let i_pc = Array2::<f64>::eye(pc_cols);

                let pc_null_projector = pc_null_projectors[pc_idx]
                    .as_ref()
                    .ok_or_else(|| {
                        EstimationError::LayoutError(format!(
                            "PC {} is missing a null-space projector required for anisotropic interaction penalties",
                            pc_name
                        ))
                    })?;

                // Project penalties into the constrained space defined by Alpha
                // S_proj = S_raw + Correction
                // where Correction accounts for the projection T -> T - M*alpha

                // Helper to extract rows of alpha corresponding to specific main effect blocks
                // M layout: [Intercept (1) | Sex (opt) | PGS_main | PC_null (opt) | PC_range]
                // We reconstruct the offsets used during M matrix construction.
                let mut current_offset = 0;
                // Intercept
                current_offset += 1;
                // Sex
                if let Some(sex_col) = sex_main.as_ref() {
                    current_offset += sex_col.ncols();
                }
                // PGS_main
                let pgs_main_offset = current_offset;
                let pgs_main_len = pgs_main.ncols();
                current_offset += pgs_main_len;
                // PC_null
                let pc_null_len = pc_null.as_ref().map_or(0, |z| z.ncols());
                let pc_null_offset = current_offset;
                current_offset += pc_null_len;
                // PC_range
                let pc_range_len = pc_range.ncols();
                let pc_range_offset = current_offset;

                // Extract Alpha blocks
                let alpha_pgs =
                    alpha.slice(s![pgs_main_offset..pgs_main_offset + pgs_main_len, ..]);
                let alpha_pc_null =
                    alpha.slice(s![pc_null_offset..pc_null_offset + pc_null_len, ..]);
                let alpha_pc_range =
                    alpha.slice(s![pc_range_offset..pc_range_offset + pc_range_len, ..]);

                // 1. PGS Wiggle Penalty (S_pgs_raw = S_pgs ⊗ I_pc)
                // The projection subtracts M_pgs * alpha_pgs.
                // M_pgs corresponds to the PGS main effect basis.
                // The penalty energy of M_pgs is calculated using s_pgs_main (already computed).
                // Correction = alpha_pgs^T * (s_pgs_main * N_pc) * alpha_pgs
                // (Note: N_pc scaling assumes sum-of-squares over PC index in the tensor metric)
                let s_pgs_raw = kronecker_product(s_pgs_penalty.as_dense(), &i_pc);
                let mut s_pgs_proj = s_pgs_raw; // Start with raw

                // Add main effect penalty contribution (Wood 2017, Section 5.6.3)
                // The penalty on the constrained basis is Z^T S_unc Z where Z is the
                // sum-to-zero transform. We compute this from the unconstrained penalty.
                let s_pgs_main_unc = diff_penalty_cache
                    .get(pgs_main_basis_unc.ncols(), config.penalty_order)?
                    .as_dense()
                    .clone();
                let s_pgs_main_constrained = pgs_z_transform
                    .t()
                    .dot(&s_pgs_main_unc.dot(&pgs_z_transform));

                // Scale by N_pc (number of columns in the PC dimension of the tensor)
                let s_pgs_main_scaled = s_pgs_main_constrained.clone() * (pc_cols as f64);
                let correction_pgs = alpha_pgs.t().dot(&s_pgs_main_scaled.dot(&alpha_pgs));

                // Subtract cross terms: alpha^T S_cross + S_cross^T alpha
                // S_cross connects M_pgs (coeff i) with T (coeffs i,j).
                // Effectively S_cross ~ S_pgs ⊗ 1^T.
                // We need to connect the CONSTRAINED main basis (rows of S_cross)
                // to the UNCONSTRAINED tensor components (cols of S_cross).
                // So S_cross_row = (Z^T S_unc) ⊗ ones(1, pc_cols).
                let s_pgs_main_mixed = pgs_z_transform.t().dot(&s_pgs_main_unc);
                let ones_pc = Array2::<f64>::ones((1, pc_cols));
                let s_cross_row = kronecker_product(&s_pgs_main_mixed, &ones_pc); // (k_pgs_con, k_pgs_unc * k_pc)
                let term2 = alpha_pgs.t().dot(&s_cross_row); // (k_tensor, k_tensor)

                // S_proj = S_raw + alpha^T S_main alpha - (alpha^T S_cross + S_cross^T alpha)
                // Note: The function is f = T*beta - M*alpha*beta.
                // J(f) = beta^T S_T beta + beta^T alpha^T S_M alpha beta - 2 beta^T S_TM alpha beta.
                // S_TM corresponds to <T, M>_J.
                // Our S_cross_row corresponds to <M, T>_J (rows are M, cols are T).
                // So term2 is alpha^T <M, T>_J. Matches.
                s_pgs_proj = s_pgs_proj + correction_pgs - term2.clone() - term2.t();

                // 2. PC Wiggle Penalty (S_pc_raw = I_pgs ⊗ S_pc)
                // Similar logic for PC parts.
                // M components: PC_null (if any) and PC_range.
                // We need the penalty on the full PC main effect basis (null + range).
                // Reconstruct PC main penalty on (null|range) basis.
                let s_pc_main_unc = diff_penalty_cache
                    .get(
                        pc_unconstrained_bases_main[pc_idx].ncols(),
                        config.penalty_order,
                    )?
                    .as_dense()
                    .clone();
                // Transform to (Null | Range) space
                // Z_total = [Z_null | Z_range]
                let z_null = &pc_null_transforms[pc_name];
                let z_range = range_transforms.get(pc_name).unwrap();
                let z_total = if z_null.ncols() > 0 {
                    let mut combined =
                        Array2::zeros((z_null.nrows(), z_null.ncols() + z_range.ncols()));
                    combined.slice_mut(s![.., 0..z_null.ncols()]).assign(z_null);
                    combined.slice_mut(s![.., z_null.ncols()..]).assign(z_range);
                    combined
                } else {
                    z_range.clone()
                };

                let s_pc_main_constrained = z_total.t().dot(&s_pc_main_unc.dot(&z_total));
                // Note: s_pc_main_constrained should be block diagonal with 0 on null part and I on range part (if whitened).
                // But we use the actual calculated values to be exact.

                // Combine alpha rows for PC (null and range)
                let alpha_pc_len = pc_null_len + pc_range_len;
                let mut alpha_pc = Array2::zeros((alpha_pc_len, alpha.ncols()));
                if pc_null_len > 0 {
                    alpha_pc
                        .slice_mut(s![0..pc_null_len, ..])
                        .assign(&alpha_pc_null);
                }
                alpha_pc
                    .slice_mut(s![pc_null_len.., ..])
                    .assign(&alpha_pc_range);

                let s_pc_raw = kronecker_product(i_pgs, s_pc_penalty.as_dense());
                let mut s_pc_proj = s_pc_raw;

                // Correction terms
                let s_pc_main_scaled = s_pc_main_constrained.clone() * (pgs_cols as f64);
                let correction_pc = alpha_pc.t().dot(&s_pc_main_scaled.dot(&alpha_pc));

                // Cross term: S_cross ~ 1^T ⊗ S_pc
                let ones_pgs = Array2::<f64>::ones((1, pgs_cols)); // (1, k_pgs)

                // Similar to PGS, we need rectangular cross term connecting constrained Main to unconstrained Tensor
                let s_pc_main_mixed = z_total.t().dot(&s_pc_main_unc);
                let s_cross_row_pc = kronecker_product(&ones_pgs, &s_pc_main_mixed);
                let term2_pc = alpha_pc.t().dot(&s_cross_row_pc);

                s_pc_proj = s_pc_proj + correction_pc - term2_pc.clone() - term2_pc.t();

                // 3. Null Penalty (S_null_raw)
                // The null penalty targets interaction terms (p*z). Main effects have zero
                // energy in this metric since they lack interaction structure. Therefore
                // projection has no effect: S_null_proj ≈ S_null_raw. We use the raw form.
                let s_null_raw = kronecker_product(&pgs_null_projector, pc_null_projector);

                // Calculate norms on the PROJECTED matrices
                let nf1 = frobenius_norm(&s_pgs_proj).max(1e-12);
                let nf2 = frobenius_norm(&s_pc_proj).max(1e-12);
                let nf3 = frobenius_norm(&s_null_raw).max(1e-12);

                let penalty_idx_pgs = tensor_block.penalty_indices[0];
                let penalty_idx_pc = tensor_block.penalty_indices[1];
                let penalty_idx_null = tensor_block.penalty_indices[2];

                // Store scaled projected matrices (Dense)
                s_list[penalty_idx_pgs] = Some(PenaltyMatrix {
                    col_range: col_range.clone(),
                    representation: PenaltyRepresentation::Dense(s_pgs_proj * (1.0 / nf1)),
                });
                s_list[penalty_idx_pc] = Some(PenaltyMatrix {
                    col_range: col_range.clone(),
                    representation: PenaltyRepresentation::Dense(s_pc_proj * (1.0 / nf2)),
                });
                s_list[penalty_idx_null] = Some(PenaltyMatrix {
                    col_range: col_range.clone(),
                    representation: PenaltyRepresentation::Dense(s_null_raw * (1.0 / nf3)),
                });
            }
        }
    }

    // Verify the final number of coefficients matches our pre-check estimate
    let n_samples = data.y.len();
    let n_coeffs = x_matrix.ncols();
    if n_coeffs > n_samples {
        log::warn!(
            "Model is over-parameterized: {} coefficients for {} samples. \
             This should have been caught by the pre-emptive check.",
            n_coeffs,
            n_samples
        );
        // Keep a lightweight condition number check as a backstop
        // This should rarely be reached due to the pre-emptive check
        match calculate_condition_number(&x_matrix) {
            Ok(cond) if cond > 1e9 => {
                log::error!(
                    "Design matrix is severely ill-conditioned (condition number > 1e9). Aborting."
                );
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: cond,
                });
            }
            Ok(_) => {}
            Err(_) => {
                log::error!("SVD failed during condition number check. Aborting.");
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }
        }
    }

    let penalty_structs: Vec<PenaltyMatrix> = s_list
        .into_iter()
        .enumerate()
        .map(|(idx, opt)| {
            opt.ok_or_else(|| {
                EstimationError::LayoutError(format!("Penalty matrix {idx} was not constructed"))
            })
        })
        .collect::<Result<_, _>>()?;

    let dense_penalties: Vec<Array2<f64>> = penalty_structs
        .iter()
        .map(|penalty| penalty.to_dense(p))
        .collect();

    Ok((
        x_matrix,
        dense_penalties,
        layout,
        sum_to_zero_constraints,
        knot_vectors,
        range_transforms,
        pc_null_transforms,
        interaction_centering_means,
        interaction_orth_alpha,
        penalty_structs,
    ))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::create_difference_penalty_matrix;
    use crate::construction::TrainingData;
    use crate::estimate::train_model;
    use crate::model::{
        BasisConfig, InteractionPenaltyKind, LinkFunction, ModelConfig, ModelFamily,
        PrincipalComponentConfig, internal_construct_design_matrix, internal_flatten_coefficients,
    };
    use approx::assert_abs_diff_eq;
    use ndarray::s;
    use ndarray::{Array1, Array2, array};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::collections::HashMap;

    /// Maximum absolute difference helper for matrix comparisons.
    fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    /// Assemble the pseudoinverse from an eigen-decomposition.
    fn pseudoinverse_from_eigh(
        eigenvalues: &Array1<f64>,
        eigenvectors: &Array2<f64>,
        tolerance: f64,
    ) -> Array2<f64> {
        let p = eigenvectors.nrows();
        let mut inv_diag = Array2::zeros((p, p));
        for (idx, &ev) in eigenvalues.iter().enumerate() {
            if ev > tolerance {
                inv_diag[[idx, idx]] = 1.0 / ev;
            }
        }
        let vt = eigenvectors.t();
        eigenvectors.dot(&inv_diag.dot(&vt))
    }

    /// Helper asserting that the reparameterization preserves critical invariants.
    fn assert_reparam_invariants(result: &ReparamResult, rs_list: &[Array2<f64>], lambdas: &[f64]) {
        // --- Orthonormality ---
        let qs = &result.qs;
        let qtq = qs.t().dot(qs);
        let identity = Array2::eye(qs.ncols());
        let orth_diff = max_abs_diff(&qtq, &identity);
        assert!(
            orth_diff < 1e-10,
            "// VIOLATION: Qs must be orthonormal; max |QtQ - I| = {:.3e}",
            orth_diff
        );

        // --- Penalty reconstruction ---
        let mut expected_s = Array2::zeros((qs.ncols(), qs.ncols()));
        for (lambda_k, rs_k) in lambdas.iter().copied().zip(rs_list.iter()) {
            let rs_qs = rs_k.dot(qs);
            let transformed = rs_qs.t().dot(&rs_qs);
            expected_s.scaled_add(lambda_k, &transformed);
        }
        let s_diff = max_abs_diff(&expected_s, &result.s_transformed);
        assert!(
            s_diff < 1e-8,
            "// VIOLATION: Transformed penalty mismatch; max |Sλ - Σ λ_k QᵀS_kQ| = {:.3e}",
            s_diff
        );

        // --- Eigenvalue-based log-determinant ---
        let (eigenvalues, eigenvectors) = result
            .s_transformed
            .clone()
            .eigh(Side::Upper)
            .expect("eigh for invariants");
        let tol = eigenvalues
            .iter()
            .fold(0.0f64, |acc, &v| acc.max(v))
            .max(1e-12)
            * 1e-12;
        let log_det_from_eigs: f64 = eigenvalues
            .iter()
            .filter(|&&ev| ev > tol)
            .map(|&ev| ev.ln())
            .sum();
        assert!(
            (result.log_det - log_det_from_eigs).abs() < 1e-8,
            "// VIOLATION: log|Sλ| mismatch; expected {:.6e}, got {:.6e}",
            log_det_from_eigs,
            result.log_det
        );

        // --- det1 trace condition ---
        assert_eq!(
            result.det1.len(),
            lambdas.len(),
            "// VIOLATION: det1 length must match lambda count"
        );
        let s_lambda_plus = pseudoinverse_from_eigh(&eigenvalues, &eigenvectors, tol);
        for (idx, (lambda_k, rs_k)) in lambdas.iter().zip(rs_list.iter()).enumerate() {
            let rs_qs = rs_k.dot(qs);
            let s_k_transformed = rs_qs.t().dot(&rs_qs);
            let trace_term: f64 = s_lambda_plus.dot(&s_k_transformed).diag().iter().sum();
            let expected_det1 = lambda_k * trace_term;
            assert!(
                (result.det1[idx] - expected_det1).abs() < 1e-8,
                "// VIOLATION: det1 trace invariant failed for penalty {idx}; expected {:.6e}, got {:.6e}",
                expected_det1,
                result.det1[idx]
            );
        }
    }

    /// Construct canonical penalty square roots and smoothing parameters for stress scenarios.
    fn create_synthetic_penalty_scenario(
        scenario_name: &str,
    ) -> (Vec<Array2<f64>>, Vec<f64>, EngineLayout) {
        match scenario_name {
            "balanced" => {
                let p = 10;
                let mut rng = StdRng::seed_from_u64(0xA5A5);
                let mut make_root = |scale: f64| {
                    let mut root = Array2::zeros((4, p));
                    for i in 0..root.nrows() {
                        for j in 0..root.ncols() {
                            root[[i, j]] = scale * rng.random_range(-1.0..1.0);
                        }
                    }
                    root
                };
                let rs1 = make_root(0.8);
                let rs2 = make_root(0.9);
                let rs3 = make_root(1.1);
                let lambdas = vec![1.0, 1.0, 1.0];
                let layout = EngineLayout::external(p, lambdas.len());
                (vec![rs1, rs2, rs3], lambdas, layout)
            }
            "imbalanced" => {
                let p = 12;
                let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
                let mut make_root = |rank: usize| {
                    let mut root = Array2::zeros((rank, p));
                    for i in 0..rank {
                        for j in 0..p {
                            root[[i, j]] = rng.random_range(-0.5..0.5);
                        }
                    }
                    root
                };
                let rs1 = make_root(5);
                let rs2 = make_root(3);
                let rs3 = make_root(2);
                let lambdas = vec![1000.0, 1.0, 1.0];
                let layout = EngineLayout::external(p, lambdas.len());
                (vec![rs1, rs2, rs3], lambdas, layout)
            }
            "near_zero" => {
                let p = 6;
                let rs1 = Array2::zeros((0, p));
                let rs2 = Array2::zeros((0, p));
                let lambdas = vec![1e-8, 2e-8];
                let layout = EngineLayout::external(p, lambdas.len());
                (vec![rs1, rs2], lambdas, layout)
            }
            "high_rank" => {
                let p = 64;
                let mut rng = StdRng::seed_from_u64(0xBADDCAFE);
                let mut make_root = |rank: usize| {
                    let mut root = Array2::zeros((rank, p));
                    for i in 0..rank {
                        for j in 0..p {
                            root[[i, j]] = rng.random_range(-1.0..1.0);
                        }
                    }
                    root
                };
                let rs1 = make_root(52);
                let rs2 = make_root(55);
                let rs3 = make_root(50);
                let lambdas = vec![2.0, 1.5, 0.75];
                let layout = EngineLayout::external(p, lambdas.len());
                (vec![rs1, rs2, rs3], lambdas, layout)
            }
            "degenerate" => {
                let p = 8;
                let eigenvalues: [f64; 8] = [4.0, 4.0, 1.0, 1.0, 0.25, 0.25, 0.0, 0.0];
                let mut rs: Vec<Array2<f64>> = Vec::new();
                for &scale in &eigenvalues {
                    if scale > 0.0 {
                        let mut row = Array2::zeros((1, p));
                        let idx = rs.len();
                        row[[0, idx]] = scale.sqrt();
                        rs.push(row);
                    }
                }
                let rs_stack = {
                    let total_rows: usize = rs.iter().map(|m| m.nrows()).sum();
                    let mut stack = Array2::zeros((total_rows, p));
                    let mut offset = 0;
                    for block in rs {
                        let rows = block.nrows();
                        stack
                            .slice_mut(s![offset..offset + rows, ..])
                            .assign(&block);
                        offset += rows;
                    }
                    stack
                };
                let lambdas = vec![3.0, 0.5];
                let layout = EngineLayout::external(p, lambdas.len());
                (vec![rs_stack.clone(), rs_stack], lambdas, layout)
            }
            other => panic!("Unknown synthetic penalty scenario: {other}"),
        }
    }

    /// Generate logistic training data with controllable seeds.
    fn create_logistic_training_data(n_samples: usize, num_pcs: usize, seed: u64) -> TrainingData {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut p = Array1::zeros(n_samples);
        for val in p.iter_mut() {
            *val = rng.random_range(-2.5..2.5);
        }

        let mut pcs = Array2::zeros((n_samples, num_pcs));
        for i in 0..n_samples {
            for j in 0..num_pcs {
                pcs[[i, j]] = rng.random_range(-2.0..2.0);
            }
        }

        let mut eta = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut val = 0.6 * p[i] + rng.random_range(-0.3..0.3);
            for j in 0..num_pcs {
                let weight = 0.2 + 0.1 * (j as f64);
                val += weight * pcs[[i, j]];
            }
            eta[i] = val;
        }

        let mut y = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let prob = 1.0 / (1.0 + (-eta[i]).exp());
            y[i] = if rng.random_range(0.0..1.0) < prob {
                1.0
            } else {
                0.0
            };
        }

        let sex = Array1::from_iter((0..n_samples).map(|_| {
            if rng.random_range(0.0..1.0) < 0.5 {
                1.0
            } else {
                0.0
            }
        }));
        let weights = Array1::ones(n_samples);

        TrainingData {
            y,
            p,
            sex,
            pcs,
            weights,
        }
    }

    fn range_from_column(col: &Array1<f64>) -> (f64, f64) {
        (
            col.iter().fold(f64::INFINITY, |acc, &v| acc.min(v)),
            col.iter().fold(f64::NEG_INFINITY, |acc, &v| acc.max(v)),
        )
    }

    fn logistic_model_config(
        include_pc_mains: bool,
        include_interactions: bool,
        data: &TrainingData,
    ) -> ModelConfig {
        let (pgs_knots, pc_knots) = if include_interactions {
            (1, 1)
        } else if include_pc_mains {
            (1, 0)
        } else {
            (1, 0)
        };

        let pgs_basis_config = BasisConfig {
            num_knots: pgs_knots,
            degree: 3,
        };
        let pc_basis_template = BasisConfig {
            num_knots: pc_knots.max(1),
            degree: 3,
        };

        let pc_configs: Vec<PrincipalComponentConfig> = if include_pc_mains {
            (0..data.pcs.ncols())
                .map(|idx| {
                    let col = data.pcs.column(idx).to_owned();
                    PrincipalComponentConfig {
                        name: format!("PC{}", idx + 1),
                        basis_config: pc_basis_template.clone(),
                        range: range_from_column(&col),
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        ModelConfig {
            model_family: ModelFamily::Gam(LinkFunction::Logit),
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 15,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 0,
            firth_bias_reduction: false,
            reml_parallel_threshold: crate::model::default_reml_parallel_threshold(),
            pgs_basis_config,
            pc_configs,
            pgs_range: range_from_column(&data.p),
            interaction_penalty: if include_pc_mains && include_interactions {
                InteractionPenaltyKind::Anisotropic
            } else {
                InteractionPenaltyKind::Isotropic
            },
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
            mcmc_enabled: false,
            calibrator_enabled: false,
            survival: None,
        }
    }

    /// Helper function to create test data for construction tests
    fn create_test_data_for_construction(
        n_samples: usize,
        num_pcs: usize,
    ) -> (TrainingData, ModelConfig) {
        let mut rng = StdRng::seed_from_u64(12345);

        // Generate PGS data (standard normal)
        let p: Array1<f64> = (0..n_samples)
            .map(|_| rng.random_range(-2.0..2.0))
            .collect();

        // Generate PC data if requested
        let pcs = if num_pcs > 0 {
            let mut pc_data = Array2::zeros((n_samples, num_pcs));
            for i in 0..n_samples {
                for j in 0..num_pcs {
                    pc_data[[i, j]] = rng.random_range(-1.5..1.5);
                }
            }
            pc_data
        } else {
            Array2::zeros((n_samples, 0))
        };

        // Generate simple response (linear combination for simplicity)
        let y: Array1<f64> = (0..n_samples)
            .map(|i| {
                let mut response = 0.5 * p[i];
                for j in 0..num_pcs {
                    response += 0.3 * pcs[[i, j]];
                }
                response + rng.random_range(-0.1..0.1) // small noise
            })
            .collect();

        // Calculate ranges before moving data into struct
        let pgs_range = (
            p.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            p.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        );

        let pc_ranges: Vec<_> = (0..num_pcs)
            .map(|j| {
                let col = pcs.column(j);
                (
                    col.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                )
            })
            .collect();

        let weights = Array1::<f64>::ones(n_samples);
        let sex = Array1::from_iter((0..n_samples).map(|i| (i % 2) as f64));
        let data = TrainingData {
            y,
            p,
            sex,
            pcs,
            weights,
        };

        // Create config with known parameters - need to provide all required fields
        let pgs_basis_config = BasisConfig {
            num_knots: 5,
            degree: 3,
        };

        // Create PrincipalComponentConfig objects for each PC
        let pc_configs: Vec<_> = (0..num_pcs)
            .map(|i| {
                let pc_range = pc_ranges[i];
                let pc_basis_config = BasisConfig {
                    num_knots: 4,
                    degree: 3,
                };
                let pc_name = format!("PC{}", i + 1);
                crate::model::PrincipalComponentConfig {
                    name: pc_name,
                    basis_config: pc_basis_config,
                    range: pc_range,
                }
            })
            .collect();

        let config = ModelConfig {
            model_family: ModelFamily::Gam(LinkFunction::Identity),
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 100,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 100,
            firth_bias_reduction: false,
            reml_parallel_threshold: crate::model::default_reml_parallel_threshold(),
            pgs_basis_config,
            pc_configs,
            pgs_range,
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
            mcmc_enabled: false,
            calibrator_enabled: false,
            survival: None,
        };

        (data, config)
    }

    #[test]
    fn test_interaction_design_matrix_is_full_rank() {
        let (data, config) = create_test_data_for_construction(100, 1);
        let (x, ..) = build_design_and_penalty_matrices(&data, &config).unwrap();

        // Calculate numerical rank via SVD
        let svd = x.svd(false, false).expect("SVD failed");
        let max_s_val = svd.1.iter().fold(0.0f64, |a, &b| a.max(b));
        let rank_tol = max_s_val * 1e-12;
        let rank = svd.1.iter().filter(|&&s| s > rank_tol).count();

        // The design matrix should be full-rank by construction
        // The code cleverly builds interaction terms using main effect bases that already
        // omit intercept columns to prevent rank deficiency from the start
        assert_eq!(
            rank,
            x.ncols(),
            "The design matrix with interactions should be full-rank by construction. Rank: {}, Columns: {}",
            rank,
            x.ncols()
        );
    }

    #[test]
    fn test_interaction_term_has_correct_penalty_structure() {
        let (data, config) = create_test_data_for_construction(100, 1);
        let (
            _,
            s_list,
            layout,
            sum_to_zero_constraints,
            _,
            range_transforms,
            pc_null_transforms,
            _,
            interaction_orth_alpha,
            penalty_structs,
        ) = build_design_and_penalty_matrices(&data, &config).unwrap();

        // With null-space penalization and the sex×PGS varying coefficient:
        // PC null + PC range + three anisotropic interaction penalties + one sex×PGS penalty
        assert_eq!(s_list.len(), penalty_structs.len());
        assert_eq!(
            s_list.len(),
            6,
            "Should have exactly 6 penalty matrices: PC null, PC range, three anisotropic interaction penalties, and one sex×PGS penalty",
        );

        let interaction_block = layout
            .penalty_map
            .iter()
            .find(|b| b.term_type == TermType::Interaction)
            .expect("Interaction block not found in layout");

        assert_eq!(interaction_block.penalty_indices.len(), 3);

        let sex_pgs_block = layout
            .penalty_map
            .iter()
            .find(|b| b.term_type == TermType::SexPgsInteraction)
            .expect("Sex×PGS block not found in layout");

        assert_eq!(sex_pgs_block.penalty_indices.len(), 1);

        // Locate explicit PC range and PC null penalty blocks
        let pc_range_block = layout
            .penalty_map
            .iter()
            .find(|b| b.term_type == TermType::PcMainEffect && b.term_name == "f(PC1)")
            .expect("PC range main-effect block not found in layout");
        let pc_null_block = layout
            .penalty_map
            .iter()
            .find(|b| b.term_type == TermType::PcMainEffect && b.term_name == "f(PC1)_null")
            .expect("PC null-space block not found in layout");

        let s_pc_range = &s_list[pc_range_block.penalty_indices[0]];
        let s_pc_null = &s_list[pc_null_block.penalty_indices[0]];

        // Build expected anisotropic penalties for comparison
        let (pgs_basis_unc, _) = create_basis::<Dense>(
            data.p.view(),
            KnotSource::Generate {
                data_range: config.pgs_range,
                num_internal_knots: config.pgs_basis_config.num_knots,
            },
            config.pgs_basis_config.degree,
            BasisOptions::value(),
        )
        .expect("PGS basis construction");
        let pgs_cols = pgs_basis_unc.ncols() - 1; // drop intercept column

        let pc_config = &config.pc_configs[0];
        let (pc_basis_unc, _) = create_basis::<Dense>(
            data.pcs.column(0).view(),
            KnotSource::Generate {
                data_range: pc_config.range,
                num_internal_knots: pc_config.basis_config.num_knots,
            },
            pc_config.basis_config.degree,
            BasisOptions::value(),
        )
        .expect("PC basis construction");
        let pc_cols = pc_basis_unc.ncols() - 1; // drop intercept column

        let s_pgs = create_difference_penalty_matrix(pgs_cols, config.penalty_order, None)
            .expect("valid PGS penalty");
        let s_pc = create_difference_penalty_matrix(pc_cols, config.penalty_order, None)
            .expect("valid PC penalty");
        let i_pgs = Array2::<f64>::eye(pgs_cols);
        let i_pc = Array2::<f64>::eye(pc_cols);

        let expected_pgs_raw = kronecker_product(&s_pgs, &i_pc);
        let expected_pc_raw = kronecker_product(&i_pgs, &s_pc);
        let (z_null_pgs, _) = basis::null_range_whiten(&s_pgs).expect("PGS null/range");
        let (z_null_pc, _) = basis::null_range_whiten(&s_pc).expect("PC null/range");
        let expected_null = kronecker_product(
            &z_null_pgs.dot(&z_null_pgs.t()),
            &z_null_pc.dot(&z_null_pc.t()),
        );

        // Reproduce the same projection used in build_design_and_penalty_matrices.
        // For T_proj = T_raw - M*alpha, the roughness penalty expands as:
        // J(T_proj β) = βᵀ S_raw β + βᵀ alphaᵀ S_main alpha β - 2 βᵀ (S_cross alpha) β.
        // This matches the S_raw + correction - cross_terms construction in production.
        let pgs_z_transform = sum_to_zero_constraints
            .get("pgs_main")
            .expect("pgs_main transform missing");
        let pc_name = &pc_config.name;
        let interaction_key = format!("f(PGS,{})", pc_name);
        let alpha = interaction_orth_alpha
            .get(&interaction_key)
            .expect("interaction alpha missing");

        let mut current_offset = 0usize;
        current_offset += 1; // intercept
        if layout.sex_col.is_some() {
            current_offset += 1;
        }
        let pgs_main_offset = current_offset;
        let pgs_main_len = pgs_z_transform.ncols();
        current_offset += pgs_main_len;
        let pc_null_len = pc_null_transforms
            .get(pc_name)
            .map(|z| z.ncols())
            .unwrap_or(0);
        let pc_null_offset = current_offset;
        current_offset += pc_null_len;
        let pc_range_len = range_transforms
            .get(pc_name)
            .expect("pc range transform missing")
            .ncols();
        let pc_range_offset = current_offset;

        let alpha_pgs = alpha.slice(s![pgs_main_offset..pgs_main_offset + pgs_main_len, ..]);
        let alpha_pc_null = if pc_null_len > 0 {
            alpha.slice(s![pc_null_offset..pc_null_offset + pc_null_len, ..])
        } else {
            alpha.slice(s![0..0, ..])
        };
        let alpha_pc_range = alpha.slice(s![pc_range_offset..pc_range_offset + pc_range_len, ..]);

        // PGS projected penalty
        let s_pgs_main_unc = create_difference_penalty_matrix(pgs_cols, config.penalty_order, None)
            .expect("PGS main penalty");
        let s_pgs_main_constrained = pgs_z_transform
            .t()
            .dot(&s_pgs_main_unc.dot(pgs_z_transform));
        let s_pgs_main_scaled = s_pgs_main_constrained * (pc_cols as f64);
        let correction_pgs = alpha_pgs.t().dot(&s_pgs_main_scaled.dot(&alpha_pgs));
        let s_pgs_main_mixed = pgs_z_transform.t().dot(&s_pgs_main_unc);
        let ones_pc = Array2::<f64>::ones((1, pc_cols));
        let s_cross_row = kronecker_product(&s_pgs_main_mixed, &ones_pc);
        let term2 = alpha_pgs.t().dot(&s_cross_row);
        let expected_pgs_proj = &expected_pgs_raw + &correction_pgs - &term2 - &term2.t();

        // PC projected penalty
        let pc_null_transform = pc_null_transforms
            .get(pc_name)
            .cloned()
            .unwrap_or_else(|| Array2::zeros((pc_cols, 0)));
        let pc_range_transform = range_transforms
            .get(pc_name)
            .expect("pc range transform missing");
        let z_total = if pc_null_transform.ncols() > 0 {
            let mut combined = Array2::zeros((
                pc_null_transform.nrows(),
                pc_null_transform.ncols() + pc_range_transform.ncols(),
            ));
            combined
                .slice_mut(s![.., 0..pc_null_transform.ncols()])
                .assign(&pc_null_transform);
            combined
                .slice_mut(s![.., pc_null_transform.ncols()..])
                .assign(pc_range_transform);
            combined
        } else {
            pc_range_transform.clone()
        };
        let s_pc_main_unc =
            create_difference_penalty_matrix(pc_basis_unc.ncols() - 1, config.penalty_order, None)
                .expect("PC main penalty");
        let s_pc_main_constrained = z_total.t().dot(&s_pc_main_unc.dot(&z_total));
        let mut alpha_pc = Array2::zeros((pc_null_len + pc_range_len, alpha.ncols()));
        if pc_null_len > 0 {
            alpha_pc
                .slice_mut(s![0..pc_null_len, ..])
                .assign(&alpha_pc_null);
        }
        alpha_pc
            .slice_mut(s![pc_null_len.., ..])
            .assign(&alpha_pc_range);
        let s_pc_main_scaled = s_pc_main_constrained * (pgs_cols as f64);
        let correction_pc = alpha_pc.t().dot(&s_pc_main_scaled.dot(&alpha_pc));
        let s_pc_main_mixed = z_total.t().dot(&s_pc_main_unc);
        let ones_pgs = Array2::<f64>::ones((1, pgs_cols));
        let s_cross_row_pc = kronecker_product(&ones_pgs, &s_pc_main_mixed);
        let term2_pc = alpha_pc.t().dot(&s_cross_row_pc);
        let expected_pc_proj = &expected_pc_raw + &correction_pc - &term2_pc - &term2_pc.t();

        let frob = |m: &Array2<f64>| m.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-12);
        let expected_pgs_norm = &expected_pgs_proj / frob(&expected_pgs_proj);
        let expected_pc_norm = &expected_pc_proj / frob(&expected_pc_proj);
        let expected_null_norm = &expected_null / frob(&expected_null);

        let s_interaction_pgs = &s_list[interaction_block.penalty_indices[0]];
        let s_interaction_pc = &s_list[interaction_block.penalty_indices[1]];
        let s_interaction_null = &s_list[interaction_block.penalty_indices[2]];

        // Check interaction penalty blocks match expected anisotropic structure
        let col_range = interaction_block.col_range.clone();
        for (row_offset, r) in col_range.clone().enumerate() {
            for (col_offset, c) in col_range.clone().enumerate() {
                assert_abs_diff_eq!(
                    s_interaction_pgs[[r, c]],
                    expected_pgs_norm[[row_offset, col_offset]],
                    epsilon = 1e-12
                );
                assert_abs_diff_eq!(
                    s_interaction_pc[[r, c]],
                    expected_pc_norm[[row_offset, col_offset]],
                    epsilon = 1e-12
                );
                assert_abs_diff_eq!(
                    s_interaction_null[[r, c]],
                    expected_null_norm[[row_offset, col_offset]],
                    epsilon = 1e-12
                );
            }
        }

        // Outside the interaction block should remain zero
        for r in 0..layout.total_coeffs {
            for c in 0..layout.total_coeffs {
                if !(col_range.contains(&r) && col_range.contains(&c)) {
                    assert_abs_diff_eq!(s_interaction_pgs[[r, c]], 0.0, epsilon = 1e-12);
                    assert_abs_diff_eq!(s_interaction_pc[[r, c]], 0.0, epsilon = 1e-12);
                    assert_abs_diff_eq!(s_interaction_null[[r, c]], 0.0, epsilon = 1e-12);
                }
            }
        }

        // Validate sex×PGS penalty structure (wiggle only)
        let (pgs_basis_unc, _) = create_basis::<Dense>(
            data.p.view(),
            KnotSource::Generate {
                data_range: config.pgs_range,
                num_internal_knots: config.pgs_basis_config.num_knots,
            },
            config.pgs_basis_config.degree,
            BasisOptions::value(),
        )
        .expect("PGS basis construction");
        let pgs_main_basis_unc = pgs_basis_unc.slice(s![.., 1..]);
        let (_, z_transform) = basis::apply_sum_to_zero_constraint(
            pgs_main_basis_unc.view(),
            Some(data.weights.view()),
        )
        .expect("sum-to-zero transform");
        let s_pgs = create_difference_penalty_matrix(
            pgs_main_basis_unc.ncols(),
            config.penalty_order,
            None,
        )
        .expect("PGS penalty");
        let s_sex_pgs_wiggle = z_transform.t().dot(&s_pgs.dot(&z_transform));
        let s_sex_pgs_wiggle_block = &s_list[sex_pgs_block.penalty_indices[0]];
        let sex_range = sex_pgs_block.col_range.clone();

        let frob = |m: &Array2<f64>| m.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-12);
        let expected_wiggle = &s_sex_pgs_wiggle / frob(&s_sex_pgs_wiggle);

        for (row_offset, r) in sex_range.clone().enumerate() {
            for (col_offset, c) in sex_range.clone().enumerate() {
                assert_abs_diff_eq!(
                    s_sex_pgs_wiggle_block[[r, c]],
                    expected_wiggle[[row_offset, col_offset]],
                    epsilon = 1e-12
                );
            }
        }

        for r in 0..layout.total_coeffs {
            for c in 0..layout.total_coeffs {
                if !(sex_range.contains(&r) && sex_range.contains(&c)) {
                    assert_abs_diff_eq!(s_sex_pgs_wiggle_block[[r, c]], 0.0, epsilon = 1e-12);
                }
            }
        }

        // Check that PC main penalty matrix has identity on PC main blocks and zeros elsewhere
        let alpha_pc_range = {
            let m = pc_range_block.col_range.len() as f64;
            if m > 0.0 { 1.0 / m.sqrt() } else { 1.0 }
        };
        for r in 0..layout.total_coeffs {
            for c in 0..layout.total_coeffs {
                if pc_range_block.col_range.contains(&r) && pc_range_block.col_range.contains(&c) {
                    if r == c {
                        assert_abs_diff_eq!(s_pc_range[[r, c]], alpha_pc_range, epsilon = 1e-12);
                    } else {
                        assert_abs_diff_eq!(s_pc_range[[r, c]], 0.0, epsilon = 1e-12);
                    }
                } else {
                    assert_abs_diff_eq!(s_pc_range[[r, c]], 0.0, epsilon = 1e-12);
                }
            }
        }

        // Check null block identity structure
        let alpha_pc_null = {
            let m = pc_null_block.col_range.len() as f64;
            if m > 0.0 { 1.0 / m.sqrt() } else { 1.0 }
        };
        for r in 0..layout.total_coeffs {
            for c in 0..layout.total_coeffs {
                if pc_null_block.col_range.contains(&r) && pc_null_block.col_range.contains(&c) {
                    if r == c {
                        assert_abs_diff_eq!(s_pc_null[[r, c]], alpha_pc_null, epsilon = 1e-12);
                    } else {
                        assert_abs_diff_eq!(s_pc_null[[r, c]], 0.0, epsilon = 1e-12);
                    }
                } else {
                    assert_abs_diff_eq!(s_pc_null[[r, c]], 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn penalty_square_roots_clean_near_singular_penalty() {
        let root = array![[1.0, 0.0, 0.0], [0.0, 1e-6, 1e-6],];
        let mut penalty = root.t().dot(&root);
        penalty[[0, 1]] += 5e-13;
        penalty[[1, 0]] -= 5e-13;

        let roots = compute_penalty_square_roots(&[penalty.clone()]).expect("square roots");
        assert_eq!(roots.len(), 1);

        let rebuilt = roots[0].t().dot(&roots[0]);
        let sanitized = sanitize_symmetric(&penalty);
        for i in 0..sanitized.nrows() {
            for j in 0..sanitized.ncols() {
                assert_abs_diff_eq!(rebuilt[[i, j]], sanitized[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn robust_eigh_clamps_small_negative_modes() {
        let mat = array![[1.0, 0.0], [0.0, -1e-13]];
        let (eigs, vectors) =
            robust_eigh(&mat, Side::Lower, "test matrix").expect("robust eigen decomposition");
        let positive_count = eigs.iter().filter(|&&val| val > 0.0).count();
        assert!(positive_count >= 1);
        assert!(eigs.iter().all(|val| *val >= 0.0));
        assert!(eigs.iter().any(|val| *val == 0.0));
        assert_eq!(vectors.nrows(), 2);
    }

    #[test]
    fn test_construction_with_no_pcs() {
        let (data, config) = create_test_data_for_construction(100, 0); // 0 PCs

        let (_, _, layout, _, _, _, _, _, _, _) =
            build_design_and_penalty_matrices(&data, &config).unwrap();

        let pgs_main_coeffs =
            config.pgs_basis_config.num_knots + config.pgs_basis_config.degree - 1;
        let expected_total_coeffs = 1 + 1 + pgs_main_coeffs + pgs_main_coeffs; // Intercept + Sex + PGS main + Sex×PGS

        assert_eq!(layout.total_coeffs, expected_total_coeffs);
        assert!(
            layout
                .penalty_map
                .iter()
                .any(|b| b.term_type == TermType::SexPgsInteraction),
            "Sex×PGS interaction block should exist in a PGS-only model."
        );
        assert_eq!(
            layout.num_penalties, 1,
            "PGS-only model should have a single penalty for the sex×PGS interaction."
        );
    }

    #[test]
    fn test_predict_linear_equals_x_beta() {
        // Build training matrices and transforms
        let (data, mut config) = create_test_data_for_construction(100, 1);
        let (
            x_training,
            _,
            layout,
            sum_to_zero_constraints,
            knot_vectors,
            range_transforms,
            pc_null_transforms,
            interaction_centering_means,
            interaction_orth_alpha,
            penalty_structs,
        ) = build_design_and_penalty_matrices(&data, &config).unwrap();
        let penalty_structs_len = penalty_structs.len();
        if penalty_structs_len == usize::MAX {
            unreachable!("penalty_structs length overflow");
        }

        // Prepare a config carrying all saved transforms
        config.sum_to_zero_constraints = sum_to_zero_constraints.clone();
        config.knot_vectors = knot_vectors.clone();
        config.range_transforms = range_transforms.clone();
        config.pc_null_transforms = pc_null_transforms.clone();
        config.interaction_centering_means = interaction_centering_means.clone();
        config.interaction_orth_alpha = interaction_orth_alpha.clone();

        use crate::model::{self, MainEffects, MappedCoefficients, TrainedModel};

        // Populate coefficients deterministically using the layout dimensions.
        let mut pcs_coeffs = HashMap::new();
        for (pc_idx, pc_config) in config.pc_configs.iter().enumerate() {
            let null_len = layout
                .pc_null_cols
                .get(pc_idx)
                .map(|range| range.len())
                .unwrap_or(0);
            let range_len = {
                let block_idx = layout.pc_main_block_idx[pc_idx];
                layout.penalty_map[block_idx].col_range.len()
            };
            let mut coeffs = Vec::with_capacity(null_len + range_len);
            for j in 0..null_len {
                coeffs.push(10.0 + (pc_idx * 10 + j) as f64);
            }
            for j in 0..range_len {
                coeffs.push(20.0 + (pc_idx * 10 + j) as f64);
            }
            pcs_coeffs.insert(pc_config.name.clone(), coeffs);
        }

        let mut pgs_coeffs = Vec::with_capacity(layout.pgs_main_cols.len());
        for j in 0..layout.pgs_main_cols.len() {
            pgs_coeffs.push(30.0 + j as f64);
        }

        let mut interaction_coeffs = HashMap::new();
        if let Some(block_idx) = layout.sex_pgs_block_idx {
            let block = &layout.penalty_map[block_idx];
            let mut coeffs = Vec::with_capacity(block.col_range.len());
            for j in 0..block.col_range.len() {
                coeffs.push(60.0 + j as f64);
            }
            interaction_coeffs.insert(block.term_name.clone(), coeffs);
        }
        for &block_idx in &layout.interaction_block_idx {
            let block = &layout.penalty_map[block_idx];
            let mut coeffs = Vec::with_capacity(block.col_range.len());
            for j in 0..block.col_range.len() {
                coeffs.push(100.0 + block_idx as f64 + j as f64);
            }
            interaction_coeffs.insert(block.term_name.clone(), coeffs);
        }

        let mapped_coeffs = MappedCoefficients {
            intercept: 0.123,
            main_effects: MainEffects {
                sex: 1.0,
                pgs: pgs_coeffs,
                pcs: pcs_coeffs,
            },
            interaction_effects: interaction_coeffs,
        };

        // Reconstruct design via the same helper used in predict_linear.
        let flattened = model::internal_flatten_coefficients(&mapped_coeffs, &config)
            .expect("failed to flatten coefficients");
        let x_reconstructed = model::internal_construct_design_matrix(
            data.p.view(),
            data.sex.view(),
            data.pcs.view(),
            &config,
            &mapped_coeffs,
        )
        .expect("failed to construct design matrix");

        assert_eq!(x_training.shape(), x_reconstructed.shape());
        let design_diff = (&x_training - &x_reconstructed)
            .mapv(|v| v.abs())
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v));
        assert!(
            design_diff < 1e-10,
            "Training and reconstructed designs must match; max |diff| = {}",
            design_diff
        );

        let model = TrainedModel {
            config: config.clone(),
            coefficients: mapped_coeffs.clone(),
            lambdas: vec![],
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
            joint_link: None,
            survival: None,
            survival_companions: HashMap::new(),
            mcmc_samples: None,
            smoothing_correction: None,
        };

        let preds_via_predict = model
            .predict_linear(data.p.view(), data.sex.view(), data.pcs.view())
            .expect("predict_linear failed");
        let preds_direct = x_training.dot(&flattened);
        let max_diff = (&preds_via_predict - &preds_direct)
            .mapv(|x| x.abs())
            .iter()
            .fold(0.0f64, |acc, &x| acc.max(x));
        assert!(
            max_diff < 1e-10,
            "Training and prediction paths must be consistent; max |diff| = {}",
            max_diff
        );
    }

    fn linspace(n: usize) -> Array1<f64> {
        if n <= 1 {
            return Array1::zeros(n);
        }
        let step = 1.0 / ((n - 1) as f64);
        Array1::from_iter((0..n).map(|i| i as f64 * step))
    }

    fn make_toy_data(n: usize) -> TrainingData {
        // Minimal, deterministic data
        let y = Array1::zeros(n);
        let p = linspace(n); // PGS
        let pc1 = linspace(n); // 1 PC
        let pcs = {
            let mut m = Array2::zeros((n, 1));
            for i in 0..n {
                m[[i, 0]] = pc1[i];
            }
            m
        };
        let weights = Array1::ones(n);
        let sex = Array1::from_iter((0..n).map(|i| (i % 2) as f64));

        TrainingData {
            y,
            p,
            sex,
            pcs,
            weights,
        }
    }

    fn frob_norm(a: &Array2<f64>) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn base_cfg() -> (BasisConfig, BasisConfig, usize) {
        // Choose cubic (degree=3) with 1 internal knot:
        // unconstrained main width = num_knots + degree = 1 + 3 = 4
        // order-2 penalty => nullspace dim = 2, whitened width = 2
        let basis = BasisConfig {
            degree: 3,
            num_knots: 1,
        };
        let penalty_order = 2;
        (basis.clone(), basis, penalty_order)
    }

    fn cfg_with_interaction(kind: InteractionPenaltyKind) -> ModelConfig {
        let (pgs_basis, pc_basis, penalty_order) = base_cfg();
        // Ranges match the toy data [0,1]
        let pc1 = crate::model::PrincipalComponentConfig {
            name: "PC1".to_string(),
            range: (0.0, 1.0),
            basis_config: pc_basis,
        };
        ModelConfig {
            model_family: ModelFamily::Gam(LinkFunction::Identity),
            penalty_order,
            convergence_tolerance: 1e-6,
            max_iterations: 100,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 100,
            firth_bias_reduction: false,
            reml_parallel_threshold: crate::model::default_reml_parallel_threshold(),
            pgs_basis_config: pgs_basis,
            pc_configs: vec![pc1],
            pgs_range: (0.0, 1.0),
            interaction_penalty: kind,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
            mcmc_enabled: false,
            calibrator_enabled: false,
            survival: None,
        }
    }

    #[test]
    fn regression_anisotropic_interaction_uses_unconstrained_width() {
        // This was the panic case before the fix: m_main = 4, order = 2.
        // Old code passed whitened rank (=2) into a 2nd-diff penalty => panic.
        let n = 40;
        let data = make_toy_data(n);
        let config = cfg_with_interaction(InteractionPenaltyKind::Anisotropic);

        let (x, s_list, layout, _, _, _, _, _, _, _) =
            build_design_and_penalty_matrices(&data, &config)
                .expect("anisotropic interaction construction should not panic");

        // There is exactly one interaction block (for PC1)
        assert_eq!(
            layout.interaction_block_idx.len(),
            1,
            "expected exactly one interaction block"
        );
        let ib = layout.interaction_block_idx[0];
        let block = &layout.penalty_map[ib];

        // Anisotropic => three penalties for the interaction (PGS wiggle, PC wiggle, joint null)
        assert_eq!(
            block.penalty_indices.len(),
            3,
            "anisotropic interaction must have three penalty indices"
        );

        // With degree=3, num_knots=1 => unconstrained main width = 4 on both margins.
        // Anisotropic interaction uses *unconstrained* widths => 4 x 4 = 16 columns.
        let expected_cols = 4 * 4;
        assert_eq!(
            block.col_range.len(),
            expected_cols,
            "anisotropic tensor block should be 16-wide (4x4) for this setup"
        );

        // Penalty sub-blocks should be present and each normalized to unit Frobenius norm
        for (idx, &pi) in block.penalty_indices.iter().enumerate() {
            let sub = s_list[pi]
                .slice(s![block.col_range.clone(), block.col_range.clone()])
                .to_owned();
            let fnorm = frob_norm(&sub);
            assert!(
                (fnorm - 1.0).abs() < 1e-8,
                "anisotropic penalty sub-block {} not unit-normalized; got {:.6e}",
                idx,
                fnorm
            );
        }

        // Sanity: design has those columns too
        assert!(
            x.ncols() >= block.col_range.end,
            "design does not contain expected interaction columns"
        );
    }

    #[test]
    fn anisotropic_penalty_is_projected_dense() {
        // Verify that Anisotropic penalties are now stored as Dense matrices (due to projection)
        // and are not just simple Kronecker products (which have structural zeros).
        let n = 40;
        let data = make_toy_data(n);
        let config = cfg_with_interaction(InteractionPenaltyKind::Anisotropic);

        // Capture penalty_structs (10th element) instead of s_list (2nd element which is dense matrices)
        let (_, _, layout, _, _, _, _, _, _, penalty_structs) =
            build_design_and_penalty_matrices(&data, &config).expect("construction success");

        let ib = layout.interaction_block_idx[0];
        let block = &layout.penalty_map[ib];
        let p_pgs_idx = block.penalty_indices[0];

        let penalty_matrix = &penalty_structs[p_pgs_idx];

        // It must be Dense now because of the projection alpha^T S alpha
        if let PenaltyRepresentation::Dense(mat) = &penalty_matrix.representation {
            // Check that it is not diagonal/banded-like (simple Kronecker of band matrices often has sparsity).
            // A projected penalty usually becomes fully dense.
            let non_zeros = mat.iter().filter(|&&x| x.abs() > 1e-12).count();
            let total = mat.len();
            let density = non_zeros as f64 / total as f64;

            // Raw penalty is S_pgs (banded) x I (diag). This is sparse.
            // Projected penalty adds alpha^T S alpha which mixes everything.
            // So density should be high.
            assert!(
                density > 0.1,
                "Projected Anisotropic penalty should be relatively dense (got density {:.3})",
                density
            );
        } else {
            panic!(
                "Anisotropic penalty should be Dense after projection, but found another representation"
            );
        }
    }

    #[test]
    fn isotropic_interaction_still_uses_whitened_width() {
        let n = 40;
        let data = make_toy_data(n);
        let config = cfg_with_interaction(InteractionPenaltyKind::Isotropic);

        let (_, s_list, layout, _, _, _, _, _, _, penalty_structs) =
            build_design_and_penalty_matrices(&data, &config)
                .expect("isotropic interaction construction should not panic");

        assert_eq!(
            layout.interaction_block_idx.len(),
            1,
            "expected exactly one interaction block"
        );
        let ib = layout.interaction_block_idx[0];
        let block = &layout.penalty_map[ib];

        // Isotropic => a single penalty for the interaction
        assert_eq!(
            block.penalty_indices.len(),
            1,
            "isotropic interaction must have exactly one penalty index"
        );

        // With degree=3, num_knots=1 => unconstrained main width = 4; order-2 nullspace dim = 2
        // Whitened range width = 4 - 2 = 2 on each margin => tensor is 2 x 2 = 4 columns.
        let expected_cols = 2 * 2;
        assert_eq!(
            block.col_range.len(),
            expected_cols,
            "isotropic tensor block should be 4-wide (2x2) for this setup"
        );

        // The single isotropic penalty sub-block is identity-like and normalized to unit Frobenius norm
        let pi = block.penalty_indices[0];
        let sub = s_list[pi]
            .slice(s![block.col_range.clone(), block.col_range.clone()])
            .to_owned();
        let fnorm = frob_norm(&sub);
        assert!(
            (fnorm - 1.0).abs() < 1e-8,
            "isotropic penalty sub-block not unit-normalized; got {:.6e}",
            fnorm
        );
        assert_eq!(s_list.len(), penalty_structs.len());
    }

    #[test]
    fn test_reparam_invariants_balanced_penalties() {
        let (rs_list, lambdas, layout) = create_synthetic_penalty_scenario("balanced");
        let reparam =
            stable_reparameterization(&rs_list, &lambdas, layout.total_coeffs).expect("balanced reparam");
        assert_reparam_invariants(&reparam, &rs_list, &lambdas);
    }

    #[test]
    fn test_reparam_invariants_imbalanced_penalties() {
        let (rs_list, lambdas, layout) = create_synthetic_penalty_scenario("imbalanced");
        let reparam = stable_reparameterization(&rs_list, &lambdas, layout.total_coeffs).expect("imbalanced");
        assert_reparam_invariants(&reparam, &rs_list, &lambdas);
    }

    #[test]
    fn test_reparam_invariants_near_zero_penalties() {
        let (rs_list, lambdas, layout) = create_synthetic_penalty_scenario("near_zero");
        let reparam = stable_reparameterization(&rs_list, &lambdas, layout.total_coeffs).expect("near_zero");
        assert_reparam_invariants(&reparam, &rs_list, &lambdas);
    }

    #[test]
    fn test_reparam_invariants_high_rank_penalties() {
        let (rs_list, lambdas, layout) = create_synthetic_penalty_scenario("high_rank");
        let reparam = stable_reparameterization(&rs_list, &lambdas, layout.total_coeffs).expect("high_rank");
        assert_reparam_invariants(&reparam, &rs_list, &lambdas);
    }

    #[test]
    fn test_reparam_invariants_degenerate_penalties() {
        let (rs_list, lambdas, layout) = create_synthetic_penalty_scenario("degenerate");
        let reparam = stable_reparameterization(&rs_list, &lambdas, layout.total_coeffs).expect("degenerate");
        assert_reparam_invariants(&reparam, &rs_list, &lambdas);
    }

    fn integration_assertions(
        data_train: &TrainingData,
        data_test: &TrainingData,
        config: &ModelConfig,
    ) {
        let trained = train_model(
            crate::estimate::TrainModelInputs {
                response: data_train.y.view(),
                primary_feature: data_train.p.view(),
                binary_feature: data_train.sex.view(),
                auxiliary_features: data_train.pcs.view(),
                sample_weight: data_train.weights.view(),
            },
            config,
        )
        .expect("training succeeds");
        let (_, s_list, layout, _, _, _, _, _, _, _) =
            build_design_and_penalty_matrices(data_train, config).expect("build matrices");
        let rs_list = compute_penalty_square_roots(&s_list).expect("square roots");
        let reparam =
            stable_reparameterization(&rs_list, &trained.lambdas, layout.total_coeffs).expect("reparam");
        assert_reparam_invariants(&reparam, &rs_list, &trained.lambdas);

        // --- Null space dimension check ---
        let mut s_lambda_original = Array2::zeros((layout.total_coeffs, layout.total_coeffs));
        for (lambda, s_k) in trained.lambdas.iter().zip(s_list.iter()) {
            s_lambda_original.scaled_add(*lambda, s_k);
        }
        let (eigs_original, _) = s_lambda_original
            .clone()
            .eigh(Side::Upper)
            .expect("eigh original");
        let eigs_original: Array1<f64> = eigs_original;
        let (eigs_transformed, _) = reparam
            .s_transformed
            .clone()
            .eigh(Side::Upper)
            .expect("eigh transformed");
        let eigs_transformed: Array1<f64> = eigs_transformed;
        let eig_tol = 1e-8;

        let null_orig = eigs_original
            .iter()
            .copied()
            .filter(|&v| v < eig_tol)
            .count();
        let null_trans = eigs_transformed
            .iter()
            .copied()
            .filter(|&v| v < eig_tol)
            .count();
        assert_eq!(
            null_orig, null_trans,
            "// VIOLATION: Null space dimension changed after reparameterization"
        );

        let log_det_direct: f64 = eigs_original
            .iter()
            .copied()
            .filter(|&v| v > eig_tol)
            .map(|v| v.ln())
            .sum();
        assert!(
            reparam.log_det.is_finite(),
            "// VIOLATION: log|Sλ| must be finite"
        );
        assert!(
            (reparam.log_det - log_det_direct).abs() < 1e-8,
            "// VIOLATION: log|Sλ| differs from eigenvalue computation"
        );

        // --- Prediction equivalence ---
        let coeffs = &trained.coefficients;
        let beta_original =
            internal_flatten_coefficients(coeffs, &trained.config).expect("flatten coefficients");
        let pcs_test = if trained.config.pc_configs.is_empty() {
            Array2::zeros((data_test.p.len(), 0))
        } else {
            data_test
                .pcs
                .slice(s![.., 0..trained.config.pc_configs.len()])
                .to_owned()
        };
        let x_test = internal_construct_design_matrix(
            data_test.p.view(),
            data_test.sex.view(),
            pcs_test.view(),
            &trained.config,
            coeffs,
        )
        .expect("construct design");
        let eta_original = x_test.dot(&beta_original);
        let qs = &reparam.qs;
        let beta_transformed = qs.t().dot(&beta_original);
        let x_transformed = x_test.dot(qs);
        let eta_transformed = x_transformed.dot(&beta_transformed);
        let max_pred_diff = eta_original
            .iter()
            .zip(eta_transformed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_pred_diff < 1e-6,
            "// VIOLATION: Predictions changed after reparameterization; max |Δη| = {:.3e}",
            max_pred_diff
        );
    }

    #[test]
    fn test_reparam_integration_pgs_only() {
        let train = create_logistic_training_data(100, 3, 11);
        let test = create_logistic_training_data(40, 3, 21);
        let config = logistic_model_config(false, false, &train);
        integration_assertions(&train, &test, &config);
    }

    #[test]
    fn test_reparam_integration_pgs_and_pc_mains() {
        let train = create_logistic_training_data(100, 3, 31);
        let test = create_logistic_training_data(40, 3, 41);
        let config = logistic_model_config(true, false, &train);
        integration_assertions(&train, &test, &config);
    }

    #[test]
    fn test_reparam_integration_full_model_with_interactions() {
        let train = create_logistic_training_data(100, 3, 51);
        let test = create_logistic_training_data(40, 3, 61);
        let config = logistic_model_config(true, true, &train);
        integration_assertions(&train, &test, &config);
    }
}
